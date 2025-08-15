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

output_path = urlparse(args.output_folder)
output_bucket_name = output_path.netloc
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
avg_jobs = []
for file_path in h5_files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # avg_file = f"{file_name.replace('.h5', '_avg.h5')}"
    # avg_outputs.append(avg_file)
    j = b.new_job(name=f"avg-{file_name}")
    avg_jobs.append(j)
    j.cloudfuse(bucket_name, '/h5_files')
    j._machine_type = config['embedding']['machine-type']
    j.storage("30G")
    
    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    j.command('git clone -b work --single-branch https://github.com/AMCalejandro/microscopy_computational_tools.git')
    j.command("cd microscopy_computational_tools")
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')
    j.command(f'pixi run python embeddings/avg_files.py --input /h5_files/{input_folder}/{file_name}.h5 --output avg.h5')

    j.command(f'mv avg.h5 {j.ofile1}')
    b.write_output(j.ofile1, f'{args.output_folder}/{file_name}_avg.h5')

# # STEP 3 — Merge averages into single file
m = b.new_job(name=f"merge-files")
for job in avg_jobs:
    m.depends_on(job)

m.cloudfuse(bucket_name, '/h5_files')
m._machine_type = config['embedding']['machine-type']
j.storage("30G")

m.command('apt update')
m.command('apt install -y git curl moreutils')
m.command('git clone -b work --single-branch https://github.com/AMCalejandro/microscopy_computational_tools.git')
m.command("cd microscopy_computational_tools")
m.command('curl -fsSL https://pixi.sh/install.sh | sh')
m.command('export PATH=/root/.pixi/bin:$PATH')
m.command('pixi install')
m.command(f'pixi run python embeddings/avg_files.py --merge --input /h5_files/{output_folder}/ --output plate_means.h5')

m.command(f'mv avg.h5 {m.ofile1}')
b.write_output(m.ofile1, f'{args.gs_path}/plate_means.h5')

b.run()