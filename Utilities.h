#pragma once

#include <fstream>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

const int MAX_FRAME_DRAWS = 2;
const int MAX_OBJECTS = 20;
const int MAX_TEXTURES = 512;  // Sampler descriptor pool (many models/materials)


const std::vector<const char* > deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME
};

struct Vertex {
	glm::vec3 color;
	glm::vec3 pos; 
	glm::vec2 tex;
	glm::vec3 normal;
};

// Max point lights (user-controlled count up to this)
const int MAX_POINT_LIGHTS = 8;
const int MAX_SPOT_LIGHTS = 4;

// Runtime-editable point light (UI writes into this)
struct UserPointLight {
	glm::vec3 position;
	glm::vec3 color;
	float intensity;
	float range;
	bool enabled;
};

// Runtime-editable spot light (konik isik: yon + ic/dis acı)
struct UserSpotLight {
	glm::vec3 position;
	glm::vec3 direction;   // normalized: isigin baktigi yon
	glm::vec3 color;
	float intensity;
	float range;
	float innerAngleDeg;   // derece, tam isik acisi
	float outerAngleDeg;   // derece, dis sinir (disinda 0)
	bool enabled;
};

// Single point light (std140: 3 x vec4 = 48 bytes)
struct PointLightUbo {
	glm::vec4 position;   // xyz = world position, w = enabled (0 or 1)
	glm::vec4 color;      // rgb = color, w = intensity
	glm::vec4 params;     // x = range (falloff distance), yzw unused
};

// Single spot light (std140: 4 x vec4 = 64 bytes)
struct SpotLightUbo {
	glm::vec4 position;   // xyz = world position, w = enabled (0 or 1)
	glm::vec4 direction;  // xyz = normalized direction (light looks this way)
	glm::vec4 color;      // rgb = color, w = intensity
	glm::vec4 params;     // x = range, y = cos(innerAngle), z = cos(outerAngle)
};

// Directional sun + point lights + spot lights (layout must match shader std140 exactly)
struct LightUbo {
	// ---- Directional (sun) ----
	glm::vec4 direction;   // normalized: from surface toward sun (fallback when starPosition.w == 0)
	glm::vec4 color;       // sun color/tint (RGB), alpha unused
	glm::vec4 starPosition; // xyz = world position of star; w=1 => L = normalize(xyz - fragPos), w=0 => use direction
	glm::mat4 lightViewProj;  // orthographic for shadow mapping
	glm::vec4 sunParams;      // x=intensity, y=ambientStrength, z=shadowBiasScale, w=shadowDarken
	glm::vec4 shadowParams;   // x=softness, y=0 norm /1 UV /2 sert golge /3 derinlik kontrast (F5-F8)
	int numPointLights;       // 0 .. MAX_POINT_LIGHTS
	float _padPoint[3];       // pad to 16 (shader: vec3 _padPoint, then 16-byte align)
	float _padAlign[4];       // std140: pointLights array starts at offset 176
	PointLightUbo pointLights[MAX_POINT_LIGHTS];
	int numSpotLights;        // 0 .. MAX_SPOT_LIGHTS
	float _padSpot[3];        // pad to 16 before spot array
	SpotLightUbo spotLights[MAX_SPOT_LIGHTS];
};

struct QueueFamilyIndices {
	int graphicsFamily = -1;
	int presentationFamily = -1; // location of pre queue

	bool isValid() {
		return graphicsFamily >= 0 && presentationFamily >= 0;
	}


};

struct SwapChainDetails {
	VkSurfaceCapabilitiesKHR surfaceCapabilities; 
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct SwapchainImage {
	VkImage image;
	VkImageView imageView;
};

static std::vector<char> readFile(const std::string& filename) {

	std::ifstream file(filename, std::ios::binary | std::ios::ate);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open a file");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> fileBuffer(fileSize);



	file.seekg(0);

	file.read(fileBuffer.data(), fileSize);

	file.close();

	return fileBuffer;
}

static uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t allowedTypes, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if ((allowedTypes & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
}

static void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags bufferProperties, VkBuffer * buffer, VkDeviceMemory *bufferMemory)
{
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = bufferSize;
	bufferInfo.usage = bufferUsageFlags;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, buffer);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a Vertex Buffer");
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);

	VkMemoryAllocateInfo memoryAllocInfo = {};
	memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocInfo.allocationSize = memRequirements.size;
	memoryAllocInfo.memoryTypeIndex = findMemoryTypeIndex(physicalDevice,memRequirements.memoryTypeBits,bufferProperties);

	result = vkAllocateMemory(device, &memoryAllocInfo, nullptr, bufferMemory);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate a Vertex Buffer memory");
	}

	vkBindBufferMemory(device, *buffer, *bufferMemory, 0);
}

static VkCommandBuffer beginCommandBuffer(VkDevice device, VkCommandPool commandPool) {
	VkCommandBuffer CommandBuffer = {};

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &CommandBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(CommandBuffer, &beginInfo);

	return CommandBuffer;
}

static void endAndSubmitCommandBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer commandBuffer) {
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void copyBuffer(VkDevice device, VkQueue transferQueue, VkCommandPool transferCommandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize bufferSize) {
	
	VkCommandBuffer transferCommandBuffer = beginCommandBuffer(device, transferCommandPool);

	VkBufferCopy bufferCopyRegion = {};
	bufferCopyRegion.srcOffset = 0;
	bufferCopyRegion.dstOffset = 0;
	bufferCopyRegion.size = bufferSize;

	vkCmdCopyBuffer(transferCommandBuffer, srcBuffer, dstBuffer, 1, &bufferCopyRegion);

	endAndSubmitCommandBuffer(device, transferCommandPool, transferQueue, transferCommandBuffer);
}





static void copyImageBuffer(VkDevice device, VkQueue transferQueue, VkCommandPool transferCommandPool, VkBuffer srcBuffer, VkImage image, uint32_t width, uint32_t height) {
	
	
	VkCommandBuffer transferCommandBuffer = beginCommandBuffer(device, transferCommandPool);

	VkBufferImageCopy imageRegion = {};
	imageRegion.bufferOffset = 0;
	imageRegion.bufferRowLength = 0;
	imageRegion.bufferImageHeight = 0;
	imageRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageRegion.imageSubresource.mipLevel = 0;
	imageRegion.imageSubresource.baseArrayLayer = 0;
	imageRegion.imageSubresource.layerCount = 1;
	imageRegion.imageOffset = { 0,0,0 };
	imageRegion.imageExtent = { width,height,1 };

	vkCmdCopyBufferToImage(transferCommandBuffer, srcBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,&imageRegion);


	endAndSubmitCommandBuffer(device, transferCommandPool, transferQueue, transferCommandBuffer);
}

static void transitionImageLayout(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
	VkCommandBuffer commandBuffer = beginCommandBuffer(device, commandPool);


	VkImageMemoryBarrier imageMemoryBarrier = {};

	imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageMemoryBarrier.oldLayout = oldLayout;
	imageMemoryBarrier.newLayout = newLayout;
	imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageMemoryBarrier.image = image;
	imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
	imageMemoryBarrier.subresourceRange.levelCount = 1;
	imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
	imageMemoryBarrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags srcStage;
	VkPipelineStageFlags dstStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	}
	
	vkCmdPipelineBarrier(
		commandBuffer,
		srcStage, dstStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &imageMemoryBarrier
	);


	endAndSubmitCommandBuffer(device, commandPool, queue, commandBuffer);
}