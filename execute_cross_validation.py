from training.pipeline.common.data_ingestion import DataIngestionPipeline
from training.pipeline.common.data_validation import DataValidationPipeline
from training.pipeline.common.feature_engineering import FeatureEngineeringTrainingPipeline
from training.pipeline.common.feature_extraction import FeatureExtractionTrainingPipeline
from training.pipeline.cross_val.cross_val import CrossValPipeline
from training.custom_logging import info_logger

PIPELINE = "Data Ingestion Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<<")
obj = DataIngestionPipeline()
obj.main()
info_logger.info(f">>>>> {PIPELINE} completed <<<<<")

PIPELINE = "Data Validation Training Pipeline"
info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
obj = DataValidationPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = "Feature Engineernig Training Pipeline"
info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
obj = FeatureEngineeringTrainingPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")

PIPELINE = "Feature Extraction Trainig Pipeline"
info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
obj = FeatureExtractionTrainingPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")

PIPELINE = "Cross Validation Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
obj = CrossValPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

