import argparse
import os
from trainer import Trainer

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='./haru', type=str)
    parser.add_argument('--instance_prompt', default='haru', type=str)
    parser.add_argument('--output_model_name', default='haru', type=str)
    parser.add_argument('--overwrite_existing_model', action='store_true')
    parser.add_argument('--validation_prompt', default='haru', type=str)
    parser.add_argument('--base_model', default='runwayml/stable-diffusion-v1-5', type=str)
    parser.add_argument('--resolution_s', default='512', type=str)
    parser.add_argument('--n_steps', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--gradient_accumulation', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--fp16', action='store_false')
    parser.add_argument('--use_8bit_adam', action='store_false')
    parser.add_argument('--checkpointing_steps', default=100, type=int)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--validation_epochs', default=100, type=int)
    parser.add_argument('--upload_to_hub', action='store_true')
    parser.add_argument('--use_private_repo', action='store_false')
    parser.add_argument('--delete_existing_repo', action='store_true')
    parser.add_argument('--upload_to', default='', type=str)
    parser.add_argument('--remove_gpu_after_training', action='store_true')
    return parser.parse_args()

def get_image_paths(root_folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif'] # extensión de los archivos de imagen que queremos buscar
    image_paths = {} # diccionario para almacenar los nombres de la subcarpeta y las rutas de los archivos de imagen
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension.lower() in image_extensions:
                folder_name = os.path.basename(dirpath)
                if folder_name not in image_paths:
                    image_paths[folder_name] = []
                image_path = os.path.join(dirpath, filename)
                image_paths[folder_name].append(image_path)
    return image_paths

def main(args):
    hf_token = os.getenv('HF_TOKEN')
    trainer = Trainer(hf_token)
    folders = get_image_paths(args.folder)
    for person, instance_images in folders.items():
        trainer.run(
            instance_images,
            args.instance_prompt,
            person,
            args.overwrite_existing_model,
            args.validation_prompt,
            args.base_model,
            args.resolution_s,
            args.n_steps,
            args.learning_rate,
            args.gradient_accumulation,
            args.seed,
            args.fp16,
            args.use_8bit_adam,
            args.checkpointing_steps,
            args.use_wandb,
            args.validation_epochs,
            args.upload_to_hub,
            args.use_private_repo,
            args.delete_existing_repo,
            args.upload_to,
            args.remove_gpu_after_training,
        )

if __name__ == '__main__':
    args = parseargs()
    main(args)