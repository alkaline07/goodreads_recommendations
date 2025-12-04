import pytest
from unittest.mock import patch, Mock
from src.model_deployment import ModelDeployer, DeploymentConfig
import src.model_deployment as md


@pytest.fixture(autouse=True)
def no_external_init(monkeypatch):
    """Prevent aiplatform.init and cloud logging from making external calls."""
    monkeypatch.setattr(md.aiplatform, "init", lambda *a, **k: None)
    # Ensure cloud_logging client doesn't attempt network calls
    monkeypatch.setattr(md, "cloud_logging", Mock())
    yield


def make_deployer():
    config = DeploymentConfig(project_id="test-project", region="us-central1")
    return ModelDeployer(config)


def test_get_model_from_registry_no_models():
    deployer = make_deployer()

    with patch.object(md.aiplatform.Model, "list", return_value=[]) as mock_list:
        result = deployer.get_model_from_registry("nonexistent-model")
        assert result is None
        mock_list.assert_called_once()


def test_get_model_from_registry_specific_version():
    deployer = make_deployer()

    parent = Mock()
    parent.resource_name = "projects/test/locations/us-central1/models/parent-model"

    # When listing models, return a parent model
    with patch.object(md.aiplatform.Model, "list", return_value=[parent]):
        # Patch the constructor to return a Mock model for the specific version
        fake_model = Mock()
        fake_model.version_id = "v1"
        with patch.object(md.aiplatform, "Model", return_value=fake_model) as mock_model_ctor:
            result = deployer.get_model_from_registry("parent-model", version="v1")
            assert result is fake_model
            mock_model_ctor.assert_called()


def test_get_model_from_registry_default_and_latest_selection():
    deployer = make_deployer()

    parent = Mock()
    parent.resource_name = "projects/test/locations/us-central1/models/parent-model"

    # Create fake versions
    v1 = Mock()
    v1.version_id = "1"
    v1.version_aliases = ["stable"]

    v2 = Mock()
    v2.version_id = "2"
    v2.version_aliases = ["default"]

    v3 = Mock()
    v3.version_id = "3"
    v3.version_aliases = []

    parent.versioning_registry = Mock()
    parent.versioning_registry.list_versions.return_value = [v1, v2, v3]

    fake_model = Mock()
    fake_model.version_id = "2"

    class DummyModel:
        @staticmethod
        def list(*a, **k):
            return [parent]

        def __new__(cls, *a, **k):
            return fake_model

    with patch('src.model_deployment.aiplatform.Model', new=DummyModel):
        result = deployer.get_model_from_registry("parent-model")
        assert result is fake_model


def test_get_or_create_endpoint_existing():
    deployer = make_deployer()

    endpoint = Mock()
    endpoint.resource_name = "projects/test/locations/us-central1/endpoints/123"
    endpoint.display_name = "goodreads-recommendation-endpoint"

    with patch.object(md.aiplatform.Endpoint, "list", return_value=[endpoint]) as mock_list:
        got = deployer.get_or_create_endpoint()
        assert got is endpoint
        mock_list.assert_called_once()


def test_get_or_create_endpoint_creates_new():
    deployer = make_deployer()

    endpoint = Mock()
    endpoint.resource_name = "projects/test/locations/us-central1/endpoints/456"
    endpoint.display_name = "goodreads-recommendation-endpoint"

    with patch.object(md.aiplatform.Endpoint, "list", return_value=[]):
        with patch.object(md.aiplatform.Endpoint, "create", return_value=endpoint) as mock_create:
            created = deployer.get_or_create_endpoint()
            assert created is endpoint
            mock_create.assert_called_once()


def test_deploy_model_success():
    deployer = make_deployer()

    # Prepare fake model and endpoint
    model = Mock()
    model.display_name = "goodreads_boosted_tree_regressor"
    model.version_id = "1"

    endpoint = Mock()
    endpoint.resource_name = "projects/test/locations/us-central1/endpoints/789"
    endpoint.display_name = "goodreads-recommendation-endpoint"

    # No existing deployed models
    endpoint.list_models.return_value = []

    # model.deploy should return an object with resource_name
    deployed = Mock()
    deployed.resource_name = "projects/test/locations/us-central1/models/deployed/1"

    model.deploy.return_value = deployed

    result = deployer.deploy_model(model, endpoint, deployed_model_display_name="deployed_test")

    assert result["status"] == "SUCCESS"
    assert "deployed_model_id" in result
    assert result["deployed_model_id"] == deployed.resource_name


def test_deploy_model_failure():
    deployer = make_deployer()

    model = Mock()
    model.display_name = "goodreads_boosted_tree_regressor"
    model.version_id = "1"

    endpoint = Mock()
    endpoint.resource_name = "projects/test/locations/us-central1/endpoints/999"

    # Simulate deploy raising an exception
    model.deploy.side_effect = Exception("deployment failed")
    # Ensure endpoint.list_models returns a real sequence to avoid Mock truthiness issues
    endpoint.list_models.return_value = []

    result = deployer.deploy_model(model, endpoint, deployed_model_display_name="deployed_test")

    assert result["status"] == "FAILED"
    assert "error" in result
    assert "deployment failed" in result["error"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])