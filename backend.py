from flask import Flask, request, jsonify
import tempfile
import os

from pipeline_agent import create_agent_executor
from resume_analyser.analyzer import extract_data_pdf
from dotenv import load_dotenv


# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Agent backend running"})


# ---------------- MAIN API ----------------
@app.route("/query", methods=["POST"])
def query_agent():

    if "file" not in request.files:
        return jsonify({"error": "PDF file is required"}), 400

    user_question = request.form.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    pdf_file = request.files["file"]

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_file.save(tmp.name)
        pdf_path = tmp.name

    try:
        # Extract text
        pdf_text = extract_data_pdf(pdf_path)

        # Run agent
        agent_executor = create_agent_executor()

        response = agent_executor.invoke({
            "input": f"""
DOCUMENT CONTENT:
{pdf_text}

USER QUESTION:
{user_question}
"""
        })

        return jsonify({
            "answer": response["output"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
