#pragma once


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <stdexcept>
#include <vector>
#include <set>
#include <array>
#include <algorithm>
#include <iostream>
#include "Utilities.h";
#include "Mesh.h"

#include "stb_image.h";
#include "MeshModel.h"



class VulkanRender
{
public:
	VulkanRender();

	int init(GLFWwindow* newWindow);

	void updateModel(int modelId, glm::mat4 newModel);
	void updateLight(LightUbo lightData);
	int createMeshModel(std::string modelFile);
	void draw();
	void cleanup();

	glm::mat4 cam;
	~VulkanRender();

private:
	GLFWwindow* window;

	int currentFrame = 0;



	struct UboViewProjection {
		glm::mat4 projection;
		glm::mat4 view;
	} uboViewProjection;

	VkInstance instance;

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	#ifdef NDEBUG
		const bool enableValidationLayers = false;
	#else
		const bool enableValidationLayers = true;
	#endif


	struct {
		VkPhysicalDevice physicalDevice;
		VkDevice logicalDevice;
	} mainDevice;


	// validation layer
	VkDebugUtilsMessengerEXT debugMessenger;

	/// <summary>
	/// objects
	/// </summary>
	VkQueue graphicsQueue;
	VkQueue presentationQueue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	VkFormat depthFormat;

	std::vector<SwapchainImage> swapChainImages;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkImage> depthBufferImages;
	std::vector <VkDeviceMemory> depthBufferImageMemory;
	std::vector <VkImageView> depthBufferImageView;

	std::vector<VkImage> colourBufferImages;
	std::vector <VkDeviceMemory> colourBufferImageMemory;
	std::vector <VkImageView> colourBufferImageView;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSetLayout samplerSetLayout;
	VkDescriptorSetLayout inputSetLayout;
	VkDescriptorSetLayout lightSetLayout;

	VkDescriptorPool descriptorPool;
	VkDescriptorPool samplerDescriptorPool;
	VkDescriptorPool inputDescriptorPool;


	std::vector<VkDescriptorSet> descriptorSets;
	std::vector<VkDescriptorSet> samplerDescriptorSets;
	std::vector<VkDescriptorSet> inputDescriptorSets;

	std::vector<VkBuffer> vpUniformBuffer;
	std::vector<VkDeviceMemory> vpUniformBufferMemory;

	std::vector<VkBuffer> lUniformBuffer;
	std::vector<VkDeviceMemory> lUniformBufferMemory;
	LightUbo lightData;

	std::vector<VkBuffer> modelUniformBuffer;
	std::vector<VkDeviceMemory> modelUniformBufferMemory;

	VkPushConstantRange pushConstantRange;

	std::vector<MeshModel> modelList;


	VkSampler textureSampler;
	std::vector<VkImage> textureImages;
	std::vector<VkDeviceMemory> textureImageMemory;
	std::vector<VkImageView> textureImageViews;



	VkPipeline graphicsPipeline;
	VkPipeline secondPipeline;
	VkPipelineLayout pipelineLayout;
	VkPipelineLayout secondPipeLineLayout;
	VkRenderPass renderPass;

	VkCommandPool graphicsCommandPool;


	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;

	


	std::vector<VkSemaphore> imageAvailable;
	std::vector<VkSemaphore> renderFinished;
	std::vector<VkFence> drawFences;


	void createInstance();
	void createDebugMessenger();

	void getPhysicalDevice();
	std::vector<const char*> getRequiredExtensions();


	void createLogicalDevice();
	void createSurface();
	void createSwapChain();
	void createRenderPass();
	void createDescriptorSetLayout();
	void createPushConstantRange();
	void createGraphicsPipeLine();
	void createColourBufferImage();
	void createDepthBuffer();
	void createFramebuffers();
	void createCommandPool();
	void createCommandBuffers();
	void createSynchronisation();
	void createTextureSampler();
	void createLightUniformBuffers();
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();
	void createInputDescriptorSets();

	void updateUniformBuffers(uint32_t imageIndex);

	void recordCommands(uint32_t currentImage);

	void allocateDynamicBufferTransferSpace();

	bool checkInstanceExtensionSupport(std::vector<const char*> * checkExtensions);
	bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	bool checkDeviceSuitable(VkPhysicalDevice device);
	bool checkValidationLayerSupport();

	QueueFamilyIndices getQueueFamilyIndices(VkPhysicalDevice device);
	SwapChainDetails getSwapChainDetails(VkPhysicalDevice device);

	VkSurfaceFormatKHR chooseBestSurfaceFromat(const std::vector<VkSurfaceFormatKHR>& formats);
	VkPresentModeKHR chooseBestPresentationMode(const std::vector<VkPresentModeKHR> presentationModes);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities);
	VkFormat chooseSupportedFormat(const std::vector<VkFormat> &formats, VkImageTiling tiling, VkFormatFeatureFlags featureFlags);
	
	VkImage createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags useFlags, VkMemoryPropertyFlags propFlags, VkDeviceMemory *imageMemory);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
	VkShaderModule createShaderModule(const std::vector<char>& code);


	int createTextureImage(std::string fileName);
	int createTexture(std::string fileName);
	int createTextureDescriptor(VkImageView textureImage);

	

	stbi_uc* loadTextureFile(std::string fileName, int* width, int* height, VkDeviceSize* imageSize);






	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
	VkResult createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
	void destroyDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

	
};

