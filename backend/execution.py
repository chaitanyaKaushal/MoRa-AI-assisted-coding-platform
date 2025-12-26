import docker
import os
import tempfile
import json

client = docker.from_env()

def run_in_sandbox(script_content: str, image="ai-coding-sandbox"):
    """
    Writes the script to a temp file, mounts it to Docker, executes it, 
    and returns the stdout output.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the python script to the host temp directory
            host_file_path = os.path.join(temp_dir, "script.py")
            with open(host_file_path, "w") as f:
                f.write(script_content)

            # Run container with volume mount
            # We map the temp_dir (Host) to /app/sandbox (Container)
            container = client.containers.run(
                image,
                command="python /app/sandbox/script.py",
                volumes={temp_dir: {'bind': '/app/sandbox', 'mode': 'rw'}},
                working_dir="/app/sandbox",
                detach=True,
                mem_limit="128m", # Limit memory
                network_disabled=True # No internet access for safety
            )
            
            # Wait for result (with timeout logic handled by docker/app config ideally)
            result = container.wait(timeout=10) 
            logs = container.logs().decode("utf-8")
            container.remove()
            
            return logs.strip()
    except Exception as e:
        print(f"Docker Error: {e}")
        return json.dumps({"error": str(e)})

# Ensure you build the image first: `docker build -t ai-coding-sandbox ./docker`