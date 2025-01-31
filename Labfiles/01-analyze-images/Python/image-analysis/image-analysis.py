from dotenv import load_dotenv
import os
import sys
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    try:
        # Load environment variables
        load_dotenv(dotenv_path="C:/AI/mslearn-ai-vision/.env")
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Debugging: Check if keys are loaded correctly
        if not ai_endpoint or not ai_key:
            raise ValueError("Missing AI_SERVICE_ENDPOINT or AI_SERVICE_KEY in the .env file.")

        # Get image path from command-line argument or use default
        image_file = 'C:\\AI\\mslearn-ai-vision\\Labfiles\\01-analyze-images\\Python\\image-analysis\\images\\street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Verify if image file exists
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        # Read image data
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )              

        # Call AnalyzeImage function
        AnalyzeImage(image_file, image_data, cv_client)

    except Exception as ex:
        print(f"Error: {ex}")

def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Get analysis results
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ],
        )

        # Display analysis results
        if result.caption:
            print("\nCaption:")
            print(f" Caption: '{result.caption.text}' (confidence: {result.caption.confidence * 100:.2f}%)")

        if result.dense_captions:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(f" Caption: '{caption.text}' (confidence: {caption.confidence * 100:.2f}%)")

        # Add additional analysis output for tags, objects, and people if needed.

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

if __name__ == "__main__":
    main()
