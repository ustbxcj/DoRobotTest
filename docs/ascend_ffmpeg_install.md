# Compiling FFmpeg with Ascend NPU Support (CANN 8.0+)

This guide details how to compile FFmpeg with hardware acceleration support (`h264_ascend`) for Ascend 310B devices (e.g., Orange Pi AIpro, Atlas 200I DK A2) using the CANN 8.0 SDK.

The `h264_ascend` encoder allows for high-performance hardware video encoding, which is critical for data collection pipelines on the robot.

## Prerequisites

*   **Hardware**: Ascend 310B (aarch64)
*   **OS**: Ubuntu 20.04 / 22.04
*   **SDK**: CANN Toolkit 8.0+ installed
*   **Permissions**: Root/sudo access

## Step 1: Environment Setup

Ensure your CANN environment variables are sourced. This allows the compiler to locate the necessary Ascend headers and libraries.

```bash
# Source the CANN environment (default location)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Verify the environment variable is set
echo $ASCEND_TOOLKIT_HOME
# Expected output: /usr/local/Ascend/ascend-toolkit/latest
```

## Step 2: Install Build Dependencies

Install standard build tools and codec libraries required for a feature-rich FFmpeg build.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git pkg-config yasm cmake \
    libx264-dev libx265-dev libnuma-dev libfdk-aac-dev libmp3lame-dev \
    libopus-dev libvorbis-dev libvpx-dev libxvidcore-dev
```

## Step 3: Download Source Code

We use **FFmpeg 4.4.1** as it is the version compatible with the official Ascend patch.

```bash
# Create a workspace directory
mkdir -p ~/ffmpeg_ascend && cd ~/ffmpeg_ascend

# Download FFmpeg 4.4.1
wget https://ffmpeg.org/releases/ffmpeg-4.4.1.tar.gz
tar -zxvf ffmpeg-4.4.1.tar.gz

# Clone the Ascend FFmpeg Plugin repository (contains the patch)
git clone https://gitee.com/ascend/mindxsdk-referenceapps.git
```

## Step 4: Apply the Ascend Patch

The patch modifies FFmpeg source files to interface with the ACL (Ascend Computing Language) APIs.

```bash
# Navigate to the plugin directory
# Note: Path structure inside the repo may vary slightly; look for 'AscendFFmpegPlugin'
cd ~/ffmpeg_ascend/mindxsdk-referenceapps/mxVision/AscendFFmpegPlugin

# Copy the patch file to your FFmpeg source root
cp *.patch ~/ffmpeg_ascend/ffmpeg-4.4.1/

# Apply the patch
cd ~/ffmpeg_ascend/ffmpeg-4.4.1
patch -p1 < ffmpeg_ascend.patch
```

> **Note**: If the patch file is named differently (e.g., `ascend_ffmpeg.patch` or similar), adjust the command above accordingly.

## Step 5: Configure and Compile

Configure the build to include the Ascend headers and link against the Ascend libraries.

```bash
cd ~/ffmpeg_ascend/ffmpeg-4.4.1

# Configure FFmpeg
./configure \
    --prefix=/usr/local \
    --enable-shared \
    --enable-gpl \
    --enable-nonfree \
    --enable-pthreads \
    --enable-libx264 \
    --enable-libmp3lame \
    --extra-cflags="-I${ASCEND_TOOLKIT_HOME}/include -I${ASCEND_TOOLKIT_HOME}/include/acllite" \
    --extra-ldflags="-L${ASCEND_TOOLKIT_HOME}/lib64 -Wl,-rpath=${ASCEND_TOOLKIT_HOME}/lib64" \
    --disable-debug \
    --disable-doc \
    --disable-static

# Compile (adjust -j argument based on your CPU cores)
make -j$(nproc)

# Install to system
sudo make install
```

## Step 6: Verify Installation

1.  **Update Library Cache**:
    ```bash
    sudo ldconfig
    ```

2.  **Check for Encoder**:
    Run the following to confirm `h264_ascend` is available:
    ```bash
    ffmpeg -encoders | grep ascend
    ```

    **Expected Output**:
    ```text
     V..... h264_ascend          H.264/AVC (Ascend hardware acceleration)
    ```

## Usage

Once installed, you can verify functionality by encoding a dummy video:

```bash
ffmpeg -y -f lavfi -i testsrc=duration=5:size=1280x720:rate=30 \
    -c:v h264_ascend \
    -b:v 2M \
    output_test.mp4
```

The DoRobot codebase (`operating_platform/utils/video.py`) is already configured to detect and use this encoder when available.

