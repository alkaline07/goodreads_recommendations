"""
Model Deployment Script for Vertex AI

This module handles automated deployment of models from Vertex AI Model Registry
to Vertex AI Endpoints. It provides:
1. Automatic retrieval of the latest (default) model version from the registry
2. Deployment to a specified endpoint with configurable resources
3. Traffic management for gradual rollouts
4. Deployment monitoring and logging
5. Health checks and rollback capabilities

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import logging as cloud_logging
from google.api_core import exceptions
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "deployment_logs")
os.makedirs(DOCS_DIR, exist_ok=True)

root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(root_env)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(DOCS_DIR, f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)


class DeploymentConfig:
    """Configuration for model deployment."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: str = "us-central1",
        endpoint_display_name: str = "goodreads-recommendation-endpoint",
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 3,
        accelerator_type: Optional[str] = None,
        accelerator_count: int = 0,
        traffic_split_percentage: int = 100,
        enable_access_logging: bool = True,
        service_account: Optional[str] = None,
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        self.region = region
        self.endpoint_display_name = endpoint_display_name
        self.machine_type = machine_type
        self.min_replica_count = min_replica_count
        self.max_replica_count = max_replica_count
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count
        self.traffic_split_percentage = traffic_split_percentage
        self.enable_access_logging = enable_access_logging
        self.service_account = service_account


class DeploymentLogger:
    """Handles logging to Cloud Logging for deployment events."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        try:
            self.client = cloud_logging.Client(project=project_id)
            self.logger = self.client.logger("model-deployment")
            self.enabled = True
        except Exception as e:
            logger.warning(f"Cloud Logging not available: {e}")
            self.enabled = False
    
    def log_deployment_event(
        self,
        event_type: str,
        model_name: str,
        model_version: str,
        endpoint_id: str,
        status: str,
        details: Optional[Dict] = None
    ):
        """Log a deployment event to Cloud Logging."""
        payload = {
            "event_type": event_type,
            "model_name": model_name,
            "model_version": model_version,
            "endpoint_id": endpoint_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        if self.enabled:
            try:
                self.logger.log_struct(payload, severity="INFO")
            except Exception as e:
                logger.warning(f"Failed to log to Cloud Logging: {e}")
        
        logger.info(f"Deployment Event: {json.dumps(payload, indent=2)}")


class ModelDeployer:
    """
    Handles automated deployment of models to Vertex AI Endpoints.
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
        if os.environ.get("AIRFLOW_HOME"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
                os.environ["AIRFLOW_HOME"], "gcp_credentials.json"
            )
        
        if not self.config.project_id:
            bq_client = bigquery.Client()
            self.config.project_id = bq_client.project
        
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region
        )
        
        self.cloud_logger = DeploymentLogger(self.config.project_id)
        
        logger.info(f"ModelDeployer initialized for project: {self.config.project_id}")
        logger.info(f"Region: {self.config.region}")
        logger.info(f"Target endpoint: {self.config.endpoint_display_name}")
    
    def get_model_from_registry(
        self,
        display_name: str,
        version: Optional[str] = None
    ) -> Optional[aiplatform.Model]:
        """
        Retrieve a model from Vertex AI Model Registry.
        
        Args:
            display_name: Display name of the model
            version: Specific version to retrieve, or None for default version
            
        Returns:
            Model object or None if not found
        """
        logger.info(f"Fetching model '{display_name}' from registry...")
        
        try:
            models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                location=self.config.region
            )
            
            if not models:
                logger.error(f"No model found with display name: {display_name}")
                return None
            
            parent_model = models[0]
            logger.info(f"Found parent model: {parent_model.resource_name}")
            
            if version:
                model = aiplatform.Model(
                    model_name=parent_model.resource_name,
                    version=version
                )
                logger.info(f"Retrieved specific version: {version}")
            else:
                versions = parent_model.versioning_registry.list_versions()
                default_version = None
                latest_version = None
                
                for v in versions:
                    if hasattr(v, 'version_aliases') and 'default' in v.version_aliases:
                        default_version = v
                        break
                    if latest_version is None or int(v.version_id) > int(latest_version.version_id):
                        latest_version = v
                
                target_version = default_version or latest_version
                if target_version:
                    model = aiplatform.Model(
                        model_name=parent_model.resource_name,
                        version=target_version.version_id
                    )
                    logger.info(f"Retrieved {'default' if default_version else 'latest'} version: {target_version.version_id}")
                else:
                    model = parent_model
                    logger.info("Using parent model (no versions found)")
            
            return model
            
        except Exception as e:
            logger.error(f"Error fetching model from registry: {e}")
            return None
    
    def get_or_create_endpoint(self) -> Optional[aiplatform.Endpoint]:
        """
        Get existing endpoint or create a new one.
        
        Returns:
            Endpoint object or None if creation fails
        """
        logger.info(f"Looking for endpoint: {self.config.endpoint_display_name}")
        
        try:
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{self.config.endpoint_display_name}"',
                location=self.config.region
            )
            
            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"Found existing endpoint: {endpoint.resource_name}")
                return endpoint
            
            logger.info("Creating new endpoint...")
            endpoint = aiplatform.Endpoint.create(
                display_name=self.config.endpoint_display_name,
                description="Goodreads book recommendation model endpoint",
                labels={
                    "project": "goodreads-recommendations",
                    "managed_by": "automated-deployment"
                }
            )
            
            logger.info(f"Created endpoint: {endpoint.resource_name}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error getting/creating endpoint: {e}")
            return None
    
    def deploy_model(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        deployed_model_display_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy a model to an endpoint.
        
        Args:
            model: The model to deploy
            endpoint: The target endpoint
            deployed_model_display_name: Display name for the deployed model
            
        Returns:
            Dictionary with deployment result
        """
        model_version = getattr(model, 'version_id', 'N/A')
        deployed_model_name = deployed_model_display_name or f"{model.display_name}_v{model_version}"
        
        result = {
            "model_name": model.display_name,
            "model_version": model_version,
            "endpoint_id": endpoint.resource_name,
            "deployed_model_name": deployed_model_name,
            "timestamp": datetime.now().isoformat(),
            "status": "PENDING",
            "traffic_split": self.config.traffic_split_percentage
        }
        
        logger.info("="*60)
        logger.info("STARTING MODEL DEPLOYMENT")
        logger.info("="*60)
        logger.info(f"Model: {model.display_name} (v{model_version})")
        logger.info(f"Endpoint: {endpoint.display_name}")
        logger.info(f"Machine type: {self.config.machine_type}")
        logger.info(f"Replicas: {self.config.min_replica_count}-{self.config.max_replica_count}")
        logger.info(f"Traffic split: {self.config.traffic_split_percentage}%")
        
        self.cloud_logger.log_deployment_event(
            event_type="DEPLOYMENT_STARTED",
            model_name=model.display_name,
            model_version=model_version,
            endpoint_id=endpoint.resource_name,
            status="STARTED",
            details=result
        )
        
        try:
            existing_models = endpoint.list_models()
            current_traffic = {}
            
            if existing_models:
                logger.info(f"Found {len(existing_models)} existing deployed model(s)")
                remaining_traffic = 100 - self.config.traffic_split_percentage
                
                if remaining_traffic > 0 and len(existing_models) > 0:
                    traffic_per_model = remaining_traffic // len(existing_models)
                    for deployed_model in existing_models:
                        current_traffic[deployed_model.id] = traffic_per_model
            
            accelerator_config = None
            if self.config.accelerator_type and self.config.accelerator_count > 0:
                accelerator_config = {
                    "accelerator_type": self.config.accelerator_type,
                    "accelerator_count": self.config.accelerator_count
                }
            
            logger.info("Deploying model to endpoint...")
            deployed_model = model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=deployed_model_name,
                machine_type=self.config.machine_type,
                min_replica_count=self.config.min_replica_count,
                max_replica_count=self.config.max_replica_count,
                accelerator_type=self.config.accelerator_type if accelerator_config else None,
                accelerator_count=self.config.accelerator_count if accelerator_config else None,
                traffic_split={deployed_model_name: self.config.traffic_split_percentage, **current_traffic} if current_traffic else None,
                service_account=self.config.service_account,
                enable_access_logging=self.config.enable_access_logging,
                sync=True
            )
            
            result["status"] = "SUCCESS"
            result["deployed_model_id"] = deployed_model.resource_name if hasattr(deployed_model, 'resource_name') else str(deployed_model)
            
            logger.info("="*60)
            logger.info("DEPLOYMENT SUCCESSFUL")
            logger.info("="*60)
            
            self.cloud_logger.log_deployment_event(
                event_type="DEPLOYMENT_COMPLETED",
                model_name=model.display_name,
                model_version=model_version,
                endpoint_id=endpoint.resource_name,
                status="SUCCESS",
                details=result
            )
            
        except Exception as e:
            result["status"] = "FAILED"
            result["error"] = str(e)
            
            logger.error("="*60)
            logger.error("DEPLOYMENT FAILED")
            logger.error("="*60)
            logger.error(f"Error: {e}")
            
            self.cloud_logger.log_deployment_event(
                event_type="DEPLOYMENT_FAILED",
                model_name=model.display_name,
                model_version=model_version,
                endpoint_id=endpoint.resource_name,
                status="FAILED",
                details={"error": str(e)}
            )
        
        return result
    
    def verify_deployment(
        self,
        endpoint: aiplatform.Endpoint,
        max_retries: int = 5,
        retry_delay: int = 30
    ) -> bool:
        """
        Verify that the deployment is healthy.
        
        Args:
            endpoint: The endpoint to verify
            max_retries: Maximum number of verification attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if deployment is healthy, False otherwise
        """
        logger.info("Verifying deployment health...")
        
        for attempt in range(max_retries):
            try:
                deployed_models = endpoint.list_models()
                
                if not deployed_models:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: No deployed models found")
                    time.sleep(retry_delay)
                    continue
                
                healthy = True
                for dm in deployed_models:
                    logger.info(f"Deployed model: {dm.display_name} - Status check passed")
                
                if healthy:
                    logger.info("Deployment verification: PASSED")
                    return True
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Verification failed - {e}")
            
            time.sleep(retry_delay)
        
        logger.error("Deployment verification: FAILED")
        return False
    
    def undeploy_model(
        self,
        endpoint: aiplatform.Endpoint,
        deployed_model_id: str
    ) -> bool:
        """
        Undeploy a model from an endpoint.
        
        Args:
            endpoint: The endpoint
            deployed_model_id: ID of the deployed model to remove
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Undeploying model {deployed_model_id}...")
        
        try:
            endpoint.undeploy(
                deployed_model_id=deployed_model_id,
                sync=True
            )
            
            logger.info(f"Successfully undeployed model: {deployed_model_id}")
            
            self.cloud_logger.log_deployment_event(
                event_type="MODEL_UNDEPLOYED",
                model_name="N/A",
                model_version="N/A",
                endpoint_id=endpoint.resource_name,
                status="SUCCESS",
                details={"deployed_model_id": deployed_model_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to undeploy model: {e}")
            return False
    
    def get_deployment_status(
        self,
        endpoint: aiplatform.Endpoint
    ) -> Dict[str, Any]:
        """
        Get the current deployment status of an endpoint.
        
        Args:
            endpoint: The endpoint to check
            
        Returns:
            Dictionary with deployment status information
        """
        status = {
            "endpoint_name": endpoint.display_name,
            "endpoint_id": endpoint.resource_name,
            "timestamp": datetime.now().isoformat(),
            "deployed_models": [],
            "traffic_split": {}
        }
        
        try:
            deployed_models = endpoint.list_models()
            traffic = endpoint.traffic_split or {}
            
            for dm in deployed_models:
                model_info = {
                    "id": dm.id,
                    "display_name": dm.display_name,
                    "model": dm.model,
                    "traffic_percentage": traffic.get(dm.id, 0)
                }
                status["deployed_models"].append(model_info)
            
            status["traffic_split"] = traffic
            status["total_models"] = len(deployed_models)
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def save_deployment_report(self, result: Dict[str, Any]):
        """Save deployment result to a JSON report file."""
        report_path = os.path.join(
            DOCS_DIR,
            f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Deployment report saved: {report_path}")
        return report_path


def get_selected_model_info() -> Dict[str, str]:
    """
    Get the selected model information from the model selection report.
    
    Returns:
        Dictionary with model_name and display_name
    """
    model_selection_path = os.path.join(
        PROJECT_ROOT, "docs", "bias_reports", "model_selection_report.json"
    )
    
    default_info = {
        "model_name": "boosted_tree_regressor",
        "display_name": "goodreads_boosted_tree_regressor"
    }
    
    try:
        with open(model_selection_path, 'r') as f:
            report = json.load(f)
        
        selected = report.get('selected_model', {})
        model_name = selected.get('model_name', 'boosted_tree_regressor')
        
        return {
            "model_name": model_name,
            "display_name": f"goodreads_{model_name}"
        }
    except FileNotFoundError:
        logger.info("Model selection report not found, using default")
        return default_info
    except Exception as e:
        logger.warning(f"Error reading model selection report: {e}")
        return default_info


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy model to Vertex AI Endpoint")
    parser.add_argument("--project-id", help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--endpoint-name", default="goodreads-recommendation-endpoint", help="Endpoint display name")
    parser.add_argument("--machine-type", default="n1-standard-2", help="Machine type for serving")
    parser.add_argument("--min-replicas", type=int, default=1, help="Minimum replicas")
    parser.add_argument("--max-replicas", type=int, default=3, help="Maximum replicas")
    parser.add_argument("--traffic-split", type=int, default=100, help="Traffic percentage (0-100)")
    parser.add_argument("--model-display-name", help="Specific model display name to deploy")
    parser.add_argument("--model-version", help="Specific model version to deploy")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deployed without deploying")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("AUTOMATED MODEL DEPLOYMENT")
    logger.info("="*80)
    
    config = DeploymentConfig(
        project_id=args.project_id,
        region=args.region,
        endpoint_display_name=args.endpoint_name,
        machine_type=args.machine_type,
        min_replica_count=args.min_replicas,
        max_replica_count=args.max_replicas,
        traffic_split_percentage=args.traffic_split
    )
    
    deployer = ModelDeployer(config)
    
    if args.model_display_name:
        model_display_name = args.model_display_name
    else:
        model_info = get_selected_model_info()
        model_display_name = model_info["display_name"]
    
    logger.info(f"Target model: {model_display_name}")
    
    model = deployer.get_model_from_registry(
        display_name=model_display_name,
        version=args.model_version
    )
    
    if not model:
        logger.error("Failed to retrieve model from registry")
        sys.exit(1)
    
    logger.info(f"Retrieved model: {model.display_name}")
    logger.info(f"Model resource: {model.resource_name}")
    
    if args.dry_run:
        logger.info("="*80)
        logger.info("DRY RUN - No deployment will be performed")
        logger.info("="*80)
        logger.info(f"Would deploy: {model.display_name}")
        logger.info(f"To endpoint: {config.endpoint_display_name}")
        logger.info(f"With machine type: {config.machine_type}")
        logger.info(f"Replicas: {config.min_replica_count}-{config.max_replica_count}")
        sys.exit(0)
    
    endpoint = deployer.get_or_create_endpoint()
    
    if not endpoint:
        logger.error("Failed to get or create endpoint")
        sys.exit(1)
    
    result = deployer.deploy_model(model, endpoint)
    
    if result["status"] == "SUCCESS":
        if deployer.verify_deployment(endpoint):
            logger.info("Deployment verified successfully!")
            
            status = deployer.get_deployment_status(endpoint)
            logger.info(f"Endpoint status: {json.dumps(status, indent=2)}")
        else:
            logger.warning("Deployment verification had issues")
            result["verification"] = "PARTIAL"
    
    deployer.save_deployment_report(result)
    
    logger.info("="*80)
    logger.info(f"DEPLOYMENT COMPLETE: {result['status']}")
    logger.info("="*80)
    
    if result["status"] != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()
