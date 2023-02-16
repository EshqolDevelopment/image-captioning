from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import torch


class Caption(BaseModel):
    image_url: str = ''


class App:

    def __init__(self):
        self.app = FastAPI()
        self.router = APIRouter()
        self.model, self.feature_extractor, self.tokenizer, self.device, self.gen_kwargs = self.load_model()

        self.router.add_api_route('/get_captions', self.get_caption, methods=['POST'])

    @staticmethod
    def load_model():
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        return model, feature_extractor, tokenizer, device, gen_kwargs

    @staticmethod
    def convert_url_to_pil(image_url: str) -> Image:
        image_content = requests.get(url=image_url).content
        return Image.open(BytesIO(image_content))

    def predict(self, image: Image) -> str:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]

    def get_caption(self, body: Caption) -> dict:
        try:
            image = self.convert_url_to_pil(body.image_url)
            return {'caption': self.predict(image)}

        except Exception as e:
            print(e)
            return {'error': str(e)}


if __name__ == "__main__":
    api = App()
    app = FastAPI()
    app.include_router(api.router)

    uvicorn.run(app, port=80, host='0.0.0.0')
