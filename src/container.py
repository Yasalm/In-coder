from config import dockerfile, docker_cli, shared_folder, mod_dir
from loguru import logger 
from docker.errors import NotFound


def setup_base_image(dockerfile=dockerfile):
    _shared = str(shared_folder)
    img, buildg = docker_cli.images.build(dockerfile=dockerfile, rm=True, path=_shared, tag='encoder:latest')
    logger.info(f'base image built {img.short_id}')
    return img, buildg

def run_contaier(collection_name: str, model_name : str, model_type : str = None, tag : str = "encoder:latest", ):
    if '.onnx' in  model_name:
        name = f"{collection_name}-{model_name.split('.')[0]}"
    else:
        name = f"{collection_name}-{model_name}"
        logger.info(f"name of container {name}")
    container = docker_cli.containers.run(tag, detach=True, ports={8080:('127.0.0.1',None)}, environment= {"COLLECTION_NAME": collection_name, "MODEL_NAME": model_name, "MODEL_TYPE":model_type}, name=name, volumes={ str(mod_dir / collection_name): {'bind': '/app/onnx/', 'mode': 'rw'}}, )
    logger.info(f"started container {container.short_id}")
    return container

def get_container(name_or_id):
    container = None
    status = None 
    try:
        container = docker_cli.containers.get(name_or_id)
        status = container.status
    except NotFound as not_found:
        logger.debugg(f"container {name_or_id}, not not_found, error: {not_found}")
        logger.info(not_found)
    
    return container, status

def start_container(name_or_id):
    pass

def stop_container(name_or_id):
    pass

def get_containers(all=False):
    return docker_cli.containers.list(all=all)