import json
import os
import re
import sys
import subprocess

plugin_dir = os.path.dirname(__file__)
lora_dir = os.path.join(plugin_dir, 'lora')
weights_dir = os.path.join(lora_dir, 'weights')

def setup():
    install_requirements()
    append_python_paths()
    create_dirs()
    update_settings()

def install_requirements():
    requirements_path = os.path.join(plugin_dir, 'requirements.txt')
    out = subprocess.check_output(['pip', 'install', '-r', requirements_path])

    for line in out.splitlines():
        print(line)
        
def append_python_paths():
    if plugin_dir not in sys.path:
        sys.path.append(plugin_dir)

    if lora_dir not in sys.path:
        sys.path.append(lora_dir)

def create_dirs():
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

def update_settings():
    models = []

    for model in os.listdir(weights_dir):
        if model.endswith('.safetensors'):
            models.append({
                'label': re.sub(r'[-_]', ' ', model[:-12]).title(),
                'value': model
            })

    models.sort(key=lambda model: model['label'])

    settings_file = open(os.path.join(plugin_dir, 'settings.json'), 'r')
    settings_data = json.load(settings_file)   
    settings_file.close()

    settings_data['fields'][0]['groupOptions']['fields'][0]["selectOptions"]["items"] = models

    settings_file = open(os.path.join(plugin_dir, 'settings.json'), 'w')
    settings_file.write(json.dumps(settings_data, indent=4))
    settings_file.close()
