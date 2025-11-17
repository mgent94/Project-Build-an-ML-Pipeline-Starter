#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):

    # If output_description was parsed as multiple tokens, join them into a single string
    if isinstance(args.output_description, list):
        args.output_description = " ".join(args.output_description)

    run = wandb.init(project="nyc_airbnb", group="cleaning", save_code=True)
    # Convert Namespace to dict for wandb config
    run.config.update(vars(args))
    
    logger.info(f"Fetching artifact: {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    logger.info(f"Artifact fetched: {artifact}")
    logger.info(f"Artifact type: {type(artifact)}")
    logger.info(f"Downloading artifact to local path...")
    artifact_local_path = artifact.download()
    logger.info(f"Artifact downloaded to: {artifact_local_path}")
    
    # Find the CSV file in the downloaded artifact directory
    import os
    csv_files = [f for f in os.listdir(artifact_local_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in artifact at {artifact_local_path}")
    if len(csv_files) > 1:
        logger.warning(f"Multiple CSV files found, using first one: {csv_files[0]}")
    
    artifact_file = os.path.join(artifact_local_path, csv_files[0])
    logger.info(f"Reading CSV from: {artifact_file}")
    df = pd.read_csv(artifact_file)
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    # Save the cleaned file
    df.to_csv('clean_sample.csv',index=False)

    # log the new data.
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
 )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="W&B artifact reference for the input dataset (e.g. 'owner/dataset:version')",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact to create (e.g. 'clean_sample.csv')",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact (e.g. 'cleaned_data')",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        nargs="+",
        type=str,
        help="A short description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to keep when filtering outliers",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to keep when filtering outliers",
        required=True,
    )


    args = parser.parse_args()

    go(args)