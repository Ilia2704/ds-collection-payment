import logging
import json
import urllib.request

# Create a logger

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

# Check if the logger already has handlers
if not logger.hasHandlers():
    # Create a console handler and set the level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)


def send_to_teams(message: str):
    webhook_url = "https://creditzikoyi.webhook.office.com/webhookb2/44673204-40b6-440d-b5af-4c38749e0203@b2e34024-635d-40af-ae4e-81ae476f1195/IncomingWebhook/708ece0181be43e9b99ebaeb9a989ba1/fe1a8f31-cf77-4114-93a1-f1cdc0c07e05/V2Ianvqbfx2my7cOSXwJ5FlgjJCJvz7R7KbsfPLOHXGfs1"

    data = {
        "text": message
    }

    req = urllib.request.Request(
        url=webhook_url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as response:
        response_body = response.read().decode("utf-8")