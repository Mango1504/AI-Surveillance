import html
import os
import time

from config import OUTPUT_DIR, get_config


class ReportGenerator:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.config = get_config()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_html(self, incidents, session_id=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"session_{session_id or timestamp}.html")
        rows = []
        for item in incidents:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('timestamp', '')))}</td>"
                f"<td>{html.escape(str(item.get('candidate_id', '')))}</td>"
                f"<td>{html.escape(str(item.get('labels', '')))}</td>"
                f"<td>{html.escape(str(item.get('clip_path', '')))}</td>"
                "</tr>"
            )
        body = "\n".join(rows) or "<tr><td colspan='4'>No incidents recorded.</td></tr>"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Exam Proctoring Report</title>"
                "<style>body{font-family:Arial,sans-serif;margin:32px}"
                "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}"
                "th{background:#f3f4f6;text-align:left}</style></head><body>"
                f"<h1>Exam Proctoring Report</h1><p>Tier: {self.config.profile.tier}</p>"
                "<table><thead><tr><th>Time</th><th>Candidate</th><th>Labels</th><th>Clip</th></tr></thead>"
                f"<tbody>{body}</tbody></table></body></html>"
            )
        return path
