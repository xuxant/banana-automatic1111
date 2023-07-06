from potassium import Potassium, Request, Response
from webui.modules import interrogate
from webui.modules import safe
from webui.modules.api import api
import webui
import json
import torch

app = Potassium("automatic1111")

torch.load = safe.unsafe_torch_load
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

list_models = None
load_model = None

def noop(*args, **kwargs):
    pass

def unload_model():
    from modules import shared, sd_hijack, devices
    import gc
    if shared.sd_model:
        sd_hijack.model = None
        gc.collect()
        devices.torch_gc()

def register_model(model=None):
    # global model
    try:
        from modules import shared, sd_hijack
        if shared.sd_model is not model:
            unload_model()
            shared.sd_model = model
            sd_hijack.model_hijack.hijack(model)
            print("Loaded default model")
    except:
        print("Failed to hijack model.")

def load_model_by_url(url, list_models=None, load_models=None):
    # global list_models, load_model
    import webui.modules.sd_models
    import hashlib

    hash_object = hashlib.md5(url.encode())
    md5_hash = hash_object.hexdigest()

    from download_checkpoint import download
    download(url, md5_hash)

    webui.modules.sd_models.list_models = list_models
    webui.modules.sd_models.load_model = load_models

    webui.modules.sd_models.list_models()

    for m in webui.modules.sd_models.checkpoints_list.values():
        if md5_hash in m.name:
            load_model(m)
            break

    webui.modules.sd_models.list_models = noop
    webui.modules.sd_models.load_model = noop

def initialize():
    global model, list_models, load_model
    import webui.modules.sd_models

    webui.modules.sd_models.list_models()

    list_models = webui.modules.sd_models.list_models
    webui.modules.sd_models.list_models = noop

    model = webui.modules.sd_models.load_model()

    load_model = webui.modules.sd_models.load_model

    webui.modules.sd_models.list_models = noop

    register_model()

@app.init()
def init():
    
    import webui.modules.sd_models

    list_models = webui.modules.sd_models.list_models
    webui.sd_models.list_models = noop

    model = webui.modules.sd_models.load_model()
    load_model = webui.modules.sd_models.load_model

    webui.modules.sd_models.list_models = noop
    
    register_model(model=model)

    context = {
        "model": model
    }

    return context

@app.handler(route="/text2img")
def handler(context: dict, request: Request) -> Response:
    body = request.json.get("body")
    model_input = json.loads(body)
    
    params = model_input["params"]

    text_to_image = api.Api(app=app)
    response = text_to_image.text2imgapi(params)

    return Response(
        json={"output": response},
        status=200
    )


if __name__ == "__main__":
    app.serve()