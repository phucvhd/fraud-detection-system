from archives.pipelines.fraud_detection_pipeline import FraudDetectionPipeline

if __name__ == '__main__':
    pipeline = FraudDetectionPipeline()
    # pipeline.run_hyper_tune()
    pipeline.run()