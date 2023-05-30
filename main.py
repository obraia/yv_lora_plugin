import os
import torch
import numpy as np
from models.image import ImageModel
from schemas.image import ImageSchema
from schemas.image_properties import ImagePropertiesSchema
from services.sam import SamService
from utils.cv2 import Cv2Utils
from utils.time import TimeUtils
from infra.server.instance import server
from lora.utils import Lora

STATIC_FOLDER = os.path.join(os.getcwd(), 'api', 'static')
IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'images')
EMBEDDING_FOLDER = os.path.join(STATIC_FOLDER, 'embeddings')
SD_WEIGHTS_FOLDER = os.path.join(os.getcwd(), 'api', 'weights', 'sd', 'diffusers')
LORA_WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), 'lora', 'weights')

image_schema = ImageSchema()
properties_schema = ImagePropertiesSchema()

socketio = server.socketio

def handler(data, plugin_properties):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora = Lora(SD_WEIGHTS_FOLDER, LORA_WEIGHTS_FOLDER, device)
    outputs = lora.txt2img(data['properties'], plugin_properties, lambda data: socketio.emit('progress', data))

    images = []
    timestamp = TimeUtils.timestamp()
    generate_embedding = data['generate_embedding'] if 'generate_embedding' in data else False
    properties_id = properties_schema.load(data['properties']).save()

    for i, output in enumerate(outputs):
        if generate_embedding:
          embedding = SamService.generate_embedding(Cv2Utils.from_pil(output))
          embedding_name = f'{timestamp}_{i + 1}.npy'
          np.save(os.path.join(EMBEDDING_FOLDER, embedding_name), embedding)
        else:
            embedding_name = 'empty'

        image_name = f'{timestamp}_{i + 1}.png'
        output.save(os.path.join(IMAGE_FOLDER, image_name))

        image = image_schema.load({
            'image': image_name,
            'embedding': embedding_name,
            'properties_id': properties_id,
            'created_at': timestamp,
        })

        images.append(image)

    ImageModel.bulk_insert(images)

    return [image.to_json() for image in images]
