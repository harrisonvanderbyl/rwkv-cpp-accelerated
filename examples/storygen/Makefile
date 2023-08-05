VERSION=realese
SRC_DIR := ../../include/rwkv

all: nvidia amd vulkan

$(VERSION):
	@echo "\033[0;32mCreating ./$(VERSION) dir with vocab...\033[0m"
	mkdir -p $(VERSION)
	cp -r $(SRC_DIR)/tokenizer/vocab $(VERSION)/

# Rule for Nvidia GPUs
nvidia: $(VERSION)
	@echo "\033[0;32mBuilding for Nvidia GPUs...\033[0m"
	mkdir -p build
	cd build && cmake .. && cmake --build . --config $(VERSION)
	mv ./build/rwkv ./$(VERSION)/storygen-nvidia

# Rule for AMD GPUs
amd: $(VERSION)
	@echo "\033[0;32mBuilding for AMD GPUs...\033[0m"
	hipcc --std=c++17 ./storygen.cpp $(SRC_DIR)/cuda/rwkv.cu -I../../include -o ./${VERSION}/storygen-amd  #--offload-arch=gfx700,gfx701,gfx702,gfx703,gfx704,gfx705,gfx801,gfx802,gfx803,gfx805,gfx810,gfx900,gfx902,gfx904,gfx906,gfx908,gfx909,gfx1010,gfx1011,gfx1012,gfx1030

# Rule for Vulkan support
vulkan: $(VERSION) compile_shaders
	@echo "\033[0;32mBuilding with Vulkan support...\033[0m"
	g++ --std=c++17 ./storygen.cpp $(SRC_DIR)/vulkan/rwkv.cpp -I../../include -o ./${VERSION}/storygen-vulkan -lvulkan

SHADERS_SRC := $(shell find $(SRC_DIR)/vulkan/ops -name '*.comp')
SHADERS_NAMES := $(notdir $(SHADERS_SRC))
SHADERS_TARGET := $(patsubst %.comp,./$(VERSION)/%.spv,$(SHADERS_NAMES))
$(SHADERS_TARGET): _notification
	glslc $(shell find $(SRC_DIR) -name "$(notdir $(basename $@)).comp") -o $@
# Next only for cute logging
.INTERMEDIATE: _notification # Prevent from compilation on every make call
_notification:
	@echo "\033[0;32mCompiling Vulkan shaders...\033[0m"

compile_shaders: $(VERSION) $(SHADERS_TARGET)

clean:
	rm -rf build $(VERSION)
