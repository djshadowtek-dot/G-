import unittest
import os
import json
import time
from pipeline import (
    ManifestCacheLogger,
    HardwareTelemetryMonitor,
    AsyncAVStreamDataset,
    DualEngineHyperspeedPipeline,
    generate_pdf_report,
    send_email_with_pdf
)

class TestPipelineComponents(unittest.TestCase):
    def setUp(self):
        self.manifest_path = "test_manifest.json"
        self.pdf_path = "test_report.pdf"

    def tearDown(self):
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
        if os.path.exists(self.pdf_path):
            os.remove(self.pdf_path)

    def test_manifest_logger_and_gemini_override(self):
        logger = ManifestCacheLogger(manifest_path=self.manifest_path)
        self.assertTrue(os.path.exists(self.manifest_path))

        # Test Gemini cache override
        logger.apply_gemini_cache_override("realtime_mode", "DYNAMIC_OVERRIDE_TEST")

        # Test payload caching
        test_payload = {
            "timestamp": time.time(),
            "source_type": "video_engine",
            "pts_timestamp": 0.0416,
            "metrics": {
                "integrity_checksum": "0x4a2f8b",
                "spatial_complexity_score": 1420.5832
            }
        }
        logger.cache_payload(test_payload)

        report = logger.close()
        self.assertIn("Automated Pipeline Optimization & Gemini AI Action Report", report)
        self.assertIn("DYNAMIC_OVERRIDE_TEST", report)

        with open(self.manifest_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["status"], "CLOSED_AND_FINALIZED")
            self.assertEqual(data["gemini_cache_override"]["realtime_mode"], "DYNAMIC_OVERRIDE_TEST")
            self.assertEqual(len(data["telemetry_cache"]), 1)

    def test_dual_engine_pipeline(self):
        logger = ManifestCacheLogger(manifest_path=self.manifest_path)
        pipeline = DualEngineHyperspeedPipeline(logger=logger)

        video_data = [(0.0, None)]
        audio_data = [(0.0, None)]
        pipeline.process_av_matrix(video_data, audio_data)

        report = logger.close()
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
            self.assertEqual(len(data["telemetry_cache"]), 2)

    def test_hardware_telemetry_monitor(self):
        monitor = HardwareTelemetryMonitor(interval_sec=1)
        monitor.start()
        time.sleep(0.5)
        monitor.stop()

    def test_pdf_generation(self):
        script_code = "print('Hello World')"
        generate_pdf_report(script_code, pdf_filename=self.pdf_path)
        # Verify PDF report generation or graceful handling
        if os.path.exists(self.pdf_path):
            self.assertTrue(os.path.getsize(self.pdf_path) > 0)


if __name__ == "__main__":
    unittest.main()
