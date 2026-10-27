#!/usr/bin/env python3
"""
PulseHalo Smart Ring - Futuristic Activity Monitoring, Call/App Notification & ML Server
========================================================================================
Pure-Python HTTP + REST API Server serving the interactive futuristic Cyber-Biometric UI
and executing real-time Supervised ML, Unsupervised ML, and Unstructured Global Health NLP
diagnostics.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_engine import PulseHaloMLSuite, GLOBAL_COHORT_BENCHMARKS, UNSTRUCTURED_HEALTH_REPORTS

print("[PulseHalo Boot] Training Supervised (RF, MLP, LogReg, KNN, SVM) & Unsupervised (K-Means++, Isolation Forest, DBSCAN, PCA) ML Models...")
ML_SUITE = PulseHaloMLSuite()
print("[PulseHalo Ready] All ML models and Global Health Unstructured NLP Index initialized.")

PRESET_SCENARIOS = {
    "optimal": {
        "name": "Optimal Readiness (Peak Vagal Tone)",
        "vitals": {
            "heart_rate": 62, "spo2": 98.6, "hrv_rmssd": 72,
            "skin_temp": 36.5, "stress_index": 24, "steps_per_hour": 540, "sleep_hours": 7.8
        },
        "clinical_note": "Feeling well rested after 8 hours of sleep, ready for morning aerobic workout and coding session."
    },
    "stress_spike": {
        "name": "Sympathetic Stress Overload (High Cortisol)",
        "vitals": {
            "heart_rate": 98, "spo2": 96.8, "hrv_rmssd": 24,
            "skin_temp": 36.8, "stress_index": 84, "steps_per_hour": 160, "sleep_hours": 5.5
        },
        "clinical_note": "High deadline anxiety, elevated resting heart rate at desk, shallow breathing and mental fatigue."
    },
    "hypoxia_cardio": {
        "name": "Critical Cardiopulmonary / Hypoxic Warning",
        "vitals": {
            "heart_rate": 132, "spo2": 90.4, "hrv_rmssd": 17,
            "skin_temp": 37.2, "stress_index": 91, "steps_per_hour": 20, "sleep_hours": 4.9
        },
        "clinical_note": "Sudden tachycardia palpitations at rest with SpO2 oxygen desaturation below 91% and shortness of breath."
    },
    "febrile_prodrome": {
        "name": "Prodromal Viral / Febrile Immune Response",
        "vitals": {
            "heart_rate": 96, "spo2": 95.5, "hrv_rmssd": 25,
            "skin_temp": 38.1, "stress_index": 74, "steps_per_hour": 40, "sleep_hours": 6.1
        },
        "clinical_note": "Distal finger skin temperature spiked +1.5C with body chills, prodromal viral infection fatigue."
    },
    "sedentary_metabolic": {
        "name": "Sedentary Urban & Sleep-Debt Strain",
        "vitals": {
            "heart_rate": 81, "spo2": 96.6, "hrv_rmssd": 39,
            "skin_temp": 36.7, "stress_index": 58, "steps_per_hour": 28, "sleep_hours": 5.0
        },
        "clinical_note": "Sitting continuously for 3 hours with minimal steps, late night sleep debt and post-meal sluggishness."
    }
}


class PulseHaloRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_path):
        with open(html_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
            self._send_html(html_file)
        elif parsed.path == "/api/init":
            default_eval = ML_SUITE.analyze_telemetry(
                PRESET_SCENARIOS["optimal"]["vitals"],
                PRESET_SCENARIOS["optimal"]["clinical_note"]
            )
            self._send_json({
                "status": "online",
                "firmware_version": "PulseHalo-OS v4.8.2-TinyML",
                "ble_stack": "BLE 5.3 ANCS + Custom GATT (0x1850)",
                "presets": PRESET_SCENARIOS,
                "initial_analysis": default_eval,
                "global_cohorts": GLOBAL_COHORT_BENCHMARKS,
                "corpus_size": len(UNSTRUCTURED_HEALTH_REPORTS)
            })
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(raw_body)
        except Exception:
            data = {}

        if parsed.path == "/api/analyze":
            vitals = data.get("vitals", PRESET_SCENARIOS["optimal"]["vitals"])
            note = data.get("clinical_note", "")
            result = ML_SUITE.analyze_telemetry(vitals, note)
            self._send_json(result)
        else:
            self._send_json({"error": "Unknown endpoint"}, status=404)

    def log_message(self, fmt, *args):
        # Quiet clean logging
        pass


def run_server(port=8765):
    server = ThreadingHTTPServer(("0.0.0.0", port), PulseHaloRequestHandler)
    print(f"Local API server: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    run_server(port)
