import hailtop.batch as hb
import yaml
import argparse
from urllib.parse import urlparse
import subprocess
import os

parser = argparse.ArgumentParser(description='run average data on Hail Batch')
parser.add_argument('gs_path', type=str, help='GCS path to embeddings folder containing .h5 files')
parser.add_argument('output_folder', type=str, help='Output folder (GCS)')
args = parser.parse_args()

plate_path = urlparse(args.gs_path)
bucket_name = plate_path.netloc
input_folder = plate_path.path.strip('/')
output_folder = args.output_folder.rstrip('/')

# print(plate_path)
# print(bucket_name)
# print(input_folder)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(
    billing_project=config['hail-batch']['billing-project'],
    remote_tmpdir=config['hail-batch']['remote-tmpdir'],
    regions=config['hail-batch']['regions']
)
b = hb.Batch(backend=backend, name="compute_and_merge_averages")

print("Listing H5 files...")
cmd = ["gcloud", "storage", "ls", f"gs://{bucket_name}/{input_folder}/"]
h5_files = subprocess.check_output(cmd).decode().strip().split("\n")
h5_files = [f for f in h5_files if f.endswith(".h5")]
if not h5_files:
    raise ValueError(f"No .h5 files found in {args.gs_path}")

# # # STEP 2 — Create jobs for computing averages
avg_outputs = []
for file_path in h5_files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    avg_file = f"{file_name.replace('.h5', '_avg.h5')}"
    avg_outputs.append(avg_file)
    j = b.new_job(name=f"avg-{file_name}")
    j.cloudfuse(bucket_name, '/h5_files')
    j._machine_type = config['embedding']['machine-type']
    j.storage("8G")
    num_workers = config['embedding']['num-workers']

    j.command('apt update')
    j.command('apt install -y pip PyYAM')
    j.command('git clone https://github.com/AMCalejandro/microscopy_computational_tools.git')
    j.command("cd microscopy_computational_tools")
    
    j.command(f'python embeddings/avg_files.py --merge --input /h5_files/{input_folder}/{file_name} --output avg.h5')

    j.command(f'mv avg.h5 {j.ofile1}')
    b.write_output(j.ofile1, f'{output_folder}/{file_name}_avg.h5')
b.run()
# STEP 3 — Merge averages into single file
# merge_job = b.new_job(name="merge-averages")
# merge_job.image("python:3.10-slim")
# merge_job.storage("4G")
# merge_job.cpu(1)
# merge_job.memory("4G")

# merge_job.command("git clone https://github.com/AMCalejandro/microscopy_computational_tools.git")
# merge_job.command("cd microscopy_computational_tools && curl -fsSL https://pixi.sh/install.sh | sh")
# merge_job.command("export PATH=/root/.pixi/bin:$PATH")
# merge_job.command("cd microscopy_computational_tools && pixi install")

# # Download all average files
# for avg_file in avg_outputs:
#     merge_job.command(f"gcloud storage cp {args.output_folder}/{avg_file} {avg_file}")

# # Run merge script from your repo
# merge_job.command(f"cd microscopy_computational_tools && "
#                   f"export PATH=/root/.pixi/bin:$PATH && "
#                   f"pixi run python scripts/merge_avg.py {' '.join(avg_outputs)} merged_all.h5")

# # Upload merged file
# merge_job.command(f"gcloud storage cp microscopy_computational_tools/merged_all.h5 {args.output_folder}/merged_all.h5")

# # STEP 4 — Run batch
# b.run()