#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import pathlib, tempfile
import wandb

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # # Derive the base name of the file from the URL
    # basename = pathlib.Path(args.file_url).name.split("?")[0].split("#")[0]

    # # Download file, streaming so we can download files larger than
    # # the available memory. We use a named temporary file that gets
    # # destroyed at the end of the context, so we don't leave anything
    # # behind and the file gets removed even in case of errors

    # logger.info(f"Downloading {args.file_url} ...")
    # with tempfile.NamedTemporaryFile(mode='wb+', delete=False) as fp:
    #     tmp_path = fp.name    

    with wandb.init(project="nyc_airbnb", job_type="download_file") as run:
        # If artifact_description was parsed as multiple tokens, join them into a single string
        if hasattr(args, "artifact_description") and isinstance(args.artifact_description, list):
            args.artifact_description = " ".join(args.artifact_description)

        # Convert Namespace to dict for wandb config
        run.config.update(vars(args))
        
        logger.info(f"Returning sample {args.sample}")
        logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, "data", args.sample)
        
        logger.info(f"Data file path: {data_file}")
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        logger.info(f"Data file exists, size: {os.path.getsize(data_file)} bytes")
        
        log_artifact(
            args.artifact_name,
            args.artifact_type,
            args.artifact_description,
            data_file,
            run,
        )
        
        logger.info(f"Successfully uploaded artifact {args.artifact_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("--sample", type=str, required=True, help="Name of the sample to download")

    parser.add_argument("--artifact_name", type=str, required=True, help="Name for the output artifact")

    parser.add_argument("--artifact_type", type=str, required=True, help="Output artifact type.")

    parser.add_argument(
        "--artifact_description",
        nargs="+",
        required=True,
        help="A brief description of this artifact (may contain spaces)",
    )

    args = parser.parse_args()

    go(args)
