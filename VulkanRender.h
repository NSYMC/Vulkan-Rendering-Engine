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
#include "Utilities.h"
#include "Mesh.h"

#include "stb_image.h"
#include "MeshModel.h"



class VulkanRender
{
public:
	VulkanRender();

	int init(GLFWwindow* newWindow);

	void updateModel(int modelId, glm::mat4 newModel);
	/** Sahneden kaldirilan obje artik cizilmesin (sol panelden Kaldir). */
	void setModelVisible(int modelId, bool visible);
	void updateLight(LightUbo lightData);
	/* Load a model from file. If modelFile contains a path (e.g. "Scene/Sponza.gltf"),
	   textures are looked up relative to that folder first, then "Textures/". */
	int createMeshModel(std::string modelFile);
	/** Creates a 1x1 solid-color texture (e.g. for white floor). Returns texture descriptor id. */
	int createSolidColorTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
	/** Creates a single quad mesh (for sprites) and adds it to the scene. Returns model id. */
	int createQuadMeshModel(int textureId);
	/** Creates a large horizontal floor quad in XZ plane (y=0), halfSize = half extent in X and Z. Returns model id. */
	int createFloorMeshModel(float halfSize, int textureId);
	/** Creates the light sprite quad (call after loading scene so scene models stay at 0,1,...). Returns light sprite model id. */
	int createLightSpriteModel();
	/** Model id of the light sprite quad (-1 if not created). */
	int getLightSpriteModelId() const { return lightSpriteModelId; }
	/** Model id of floor/ground that should not cast shadows (-1 = none). Call after createFloorMeshModel. */
	void setFloorModelId(int id) { floorModelId = id; }
	/** When true, XYZ gizmo arrows are drawn at the light position. */
	void setLightGizmoSelected(bool selected) { lightGizmoSelected = selected; }
	bool getLightGizmoSelected() const { return lightGizmoSelected; }
	/** For picking: projection matrix (same as used for rendering). */
	glm::mat4 getProjectionMatrix() const { return uboViewProjection.projection; }
	/** Swapchain width and height. */
	void getExtent(int& outWidth, int& outHeight) const { outWidth = static_cast<int>(swapChainExtent.width); outHeight = static_cast<int>(swapChainExtent.height); }
	/** Picking: ekran koordinatindan dunya uzayinda isin (origin, direction). */
	void getPickRay(float mouseX, float mouseY, float screenW, float screenH, glm::vec3& outOrigin, glm::vec3& outDirection) const;
	/** Picking: fare konumunun near plane uzerindeki dunya koordinati (surukleme delta icin). */
	glm::vec3 getWorldPosOnNearPlane(float mouseX, float mouseY, float screenW, float screenH) const;
	/** Fare isinin verilen duzlemle kesismesi (obje derinliginde 1:1 surukleme icin). planeNormal birim vektor. */
	glm::vec3 getWorldPosOnPlane(float mouseX, float mouseY, float screenW, float screenH, const glm::vec3& planeOrigin, const glm::vec3& planeNormal) const;
	/** Kamera bakis yonu (dunya uzayi, normalize). */
	glm::vec3 getCameraForward() const;
	/** Gizmo konumu ve objenin boyutuna gore ok mesafesi/uzunlugu. objectRadius <= 0 ise 1 kullanilir. */
	void updateGizmoBuffer(glm::vec3 worldPos, float objectRadius = 1.0f);
	/** Hangi eksen tutuluyor: 0=hicbiri, 1=X, 2=Y, 3=Z (parlak cizgi icin). */
	void setGizmoHighlightAxis(int axis) { gizmoHighlightAxis = axis; }
	/** Fare ile gizmo eksenine tiklandi mi? 3D isin-segment. Donen: 0=hayir, 1=X, 2=Y, 3=Z. */
	int pickGizmoAxis(float mouseX, float mouseY, float screenW, float screenH, const glm::vec3& gizmoWorldPos) const;
	/** Ayni; view ve proj verirsen kamera konumu degisse bile tutarli calisir. */
	int pickGizmoAxis(float mouseX, float mouseY, float screenW, float screenH, const glm::vec3& gizmoWorldPos, const glm::mat4& view, const glm::mat4& proj) const;
	void draw();
	void cleanup();
	/** Yildiz kuresi: model matrisi, zaman (s), sicaklik (K). Her frame Kaynak'tan cagrilir. */
	void setStarParams(const glm::mat4& model, float timeSec, float temperatureKelvin);
	/** Gokyuzu gradient efektini ac/kapa (varsayilan: kapali; sonra tekrar acilabilir). */
	void SetSkyEnabled(bool enabled) { skyEnabled_ = enabled; }
	bool IsSkyEnabled() const { return skyEnabled_; }

	glm::mat4 cam;
	/** Shadow map looks at this world position (scene center). Set each frame in Kaynak.cpp so shadows work for any light position. */
	glm::vec3 shadowSceneCenter{ 0.0f, 0.0f, 0.0f };
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
	std::vector<bool> modelVisible;  // modelList ile ayni boyut; sahne objesi kaldirilinca false


	VkSampler textureSampler;
	std::vector<VkImage> textureImages;
	std::vector<VkDeviceMemory> textureImageMemory;
	std::vector<VkImageView> textureImageViews;



	VkPipeline graphicsPipeline;
	VkPipeline secondPipeline = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout;
	VkPipelineLayout secondPipeLineLayout = VK_NULL_HANDLE;
	VkRenderPass renderPass;

	// Shadow mapping: depth-only pass from light's view, then sample in main pass
	static const uint32_t SHADOW_MAP_SIZE = 8192;  // higher = sharper shadows (8192 = 4x more detail than 4096)
	VkFormat shadowMapDepthFormat;  // depth-only when possible (D32_SFLOAT) to avoid stencil in barriers
	VkImage shadowMapImage;
	VkDeviceMemory shadowMapMemory;
	VkImageView shadowMapImageView;
	VkRenderPass shadowMapRenderPass;
	VkFramebuffer shadowMapFramebuffer;
	VkPipeline shadowMapPipeline;
	VkPipelineLayout shadowMapPipelineLayout;
	VkDescriptorSetLayout shadowMapSetLayout;
	VkBuffer shadowMapUniformBuffer;
	VkDeviceMemory shadowMapUniformBufferMemory;
	VkDescriptorPool shadowMapDescriptorPool;
	VkDescriptorSet shadowMapDescriptorSet;
	VkSampler shadowMapSampler;

	int lightSpriteModelId = -1;
	int floorModelId = -1;
	int lightSpriteTextureId = 0;  // texture used for light sprite (set in init)
	bool lightGizmoSelected = false;
	int gizmoHighlightAxis = 0;  // 0=none, 1=X, 2=Y, 3=Z
	glm::vec3 gizmoWorldPos{ 0.f };
	float gizmoObjectRadius = 1.0f;   // secili objenin yaricapi (scale'dan); ok mesafesi bununla orantili
	float gizmoBaseOffset = 0.9f;    // ok tabaninin merkezden uzakligi = gizmoObjectRadius * 0.9
	float gizmoArrowLength = 1.2f;   // ok uzunlugu = gizmoObjectRadius * 1.2
	int gizmoArrowModelId = -1;   // UIObjects arrow model; -1 = use line gizmo
	int gizmoAxisTexId[3] = { -1, -1, -1 };  // R, G, B solid textures for X,Y,Z
	int gizmoHighlightTexId = -1;  // secili ok icin parlak sari/beyaz
	VkPipeline gizmoLinePipeline = VK_NULL_HANDLE;
	VkPipelineLayout gizmoLinePipelineLayout = VK_NULL_HANDLE;
	VkBuffer gizmoVertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory gizmoVertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer gizmoIndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory gizmoIndexBufferMemory = VK_NULL_HANDLE;
	static const float GIZMO_AXIS_LENGTH;
	static const float GIZMO_LINE_THICKNESS;  // kalinlik (dunya birimi)
	void createGizmoPipeline();

	// Sky gradient (Tier 1 Step 2)
	VkPipeline skyPipeline = VK_NULL_HANDLE;
	VkPipelineLayout skyPipelineLayout = VK_NULL_HANDLE;
	bool skyEnabled_ = false;  // gokyuzu cizimi kapali (SetSkyEnabled(true) ile acilir)
	void createSkyPipeline();

	// Yildiz simulasyonu: sicaklik (3000-30000K), blackbody renk, FBM granulation, nabiz
	VkPipeline starPipeline = VK_NULL_HANDLE;
	VkPipelineLayout starPipelineLayout = VK_NULL_HANDLE;
	VkBuffer starVertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory starVertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer starIndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory starIndexBufferMemory = VK_NULL_HANDLE;
	uint32_t starIndexCount = 0;
	void createStarPipeline();
	void createStarSphere();
	struct StarParams {
		glm::mat4 model;
		float time;
		float temperature;
		float _pad0, _pad1;
	} starParams;

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

	void createShadowMapImage();
	void createShadowMapRenderPass();
	void createShadowMapFramebuffer();
	void createShadowMapPipeline();
	void createShadowMapDescriptors();
	void updateShadowMapUniformBuffer(uint32_t imageIndex);

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


	/* createTextureImage/createTexture/loadTextureFile accept optional textureBasePath
	   so that models in Scene/ can use textures from the same folder (e.g. Scene/). */
	int createTextureImage(std::string fileName, std::string textureBasePath = "Textures/");
	int createTexture(std::string fileName, std::string textureBasePath = "Textures/");
	int createTextureDescriptor(VkImageView textureImage);

	stbi_uc* loadTextureFile(std::string fileName, int* width, int* height, VkDeviceSize* imageSize, std::string textureBasePath = "Textures/");






	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
	VkResult createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
	void destroyDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

	
};

