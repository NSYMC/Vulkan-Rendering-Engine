#include "VulkanRender.h"

const float VulkanRender::GIZMO_AXIS_LENGTH = 3.0f;
const float VulkanRender::GIZMO_LINE_THICKNESS = 0.5f;

VulkanRender::VulkanRender()
{
}

int VulkanRender::init(GLFWwindow* newWindow)
{
	window = newWindow;

	try {

		lightData = {};
		// Sun direction: from surface toward sun (normalized). Default: sun high from top-right-front
		lightData.direction = glm::vec4(glm::normalize(glm::vec3(0.4f, 0.9f, 0.3f)), 0.0f);
		lightData.color = glm::vec4(1.0f, 0.98f, 0.95f, 1.0f);  // warm white
		lightData.lightViewProj = glm::mat4(1.0f);
		lightData.sunParams = glm::vec4(1.2f, 0.35f, 1.0f, 0.15f);   // intensity, ambientStrength, shadowBiasScale, shadowDarken
		lightData.shadowParams = glm::vec4(2.5f, 0.0f, 0.0f, 0.0f); // shadowSoftness

		createInstance();
		createDebugMessenger();
		createSurface();
		getPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createRenderPass();
		createDescriptorSetLayout();
		createPushConstantRange();
		createGraphicsPipeLine();
		createColourBufferImage();
		createDepthBuffer();
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
		createUniformBuffers();
		createLightUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createInputDescriptorSets();

		createSynchronisation();
		createTextureSampler();

		std::vector<Vertex> meshVertices = {
			{{1.0,0.0,0.0},{-0.4,0.4,0.0},{1.0f,1.0f}},
			{{1.0,0.0,0.0},{-0.4,-0.4,0.0},{1.0f,0.0f} },
			{{1.0,0.0,0.0},{ 0.4,-0.4,0.0},{0.0f,0.0f}},
			{{1.0,0.0,0.0},{0.4,0.4,0.0},{0.0f,1.0f} }
		};

		std::vector<Vertex> meshVertices2 = {
			{{0.0,0.0,1.0},{-0.25,0.6,0.0},{1.0f,1.0f}},
			{{0.0,0.0,1.0},{-0.25,-0.6,0.0},{1.0f,0.0f }},
			{{0.0,0.0,1.0},{ 0.25,-0.6,0.0},{0.0f,0.0f}},
			{{0.0,0.0,1.0},{ 0.25,0.6,0.0},{0.0f,1.0f }}
		};

		std::vector<uint32_t> meshIndices = {
			0,1,2,
			2,3,0
		};




		/* Wide depth range so large scenes (e.g. Sponza) are visible far away */
		uboViewProjection.projection = glm::perspective(glm::radians(45.0f), (float)swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 3000.0f);
		//uboViewProjection.view = glm::lookAt(glm::vec3(5.0f, 2.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		uboViewProjection.view = cam;
		uboViewProjection.projection[1][1] *= -1;


		createTexture("red.png");

		lightSpriteTextureId = 0;
		try { lightSpriteTextureId = createTexture("light.png", "Sprite/"); }
		catch (...) {
			try { lightSpriteTextureId = createTexture("light.png", "Textures/"); }
			catch (...) { lightSpriteTextureId = 0; }
		}
		createGizmoPipeline();
		createSkyPipeline();
		createStarSphere();
		createStarPipeline();
		// Gizmo ok modeli (UIObjects) ve eksen renkleri
		try {
			gizmoAxisTexId[0] = createSolidColorTexture(255, 60, 60, 255);   // X kirmizi
			gizmoAxisTexId[1] = createSolidColorTexture(60, 255, 60, 255);   // Y yesil
			gizmoAxisTexId[2] = createSolidColorTexture(60, 60, 255, 255);    // Z mavi
			gizmoHighlightTexId = createSolidColorTexture(255, 255, 200, 255); // secili ok: parlak sari
			gizmoArrowModelId = createMeshModel("UIObjects/scene.gltf");
		} catch (const std::exception& e) {
			std::cerr << "Gizmo arrow load failed, using line gizmo: " << e.what() << std::endl;
			gizmoArrowModelId = -1;
		}
	}
	catch (const std::runtime_error& e) {
		printf("Error; %s \n", e);
		return EXIT_FAILURE;
	}


	return 0;
}

void VulkanRender::updateModel(int modelId, glm::mat4 newModel)
{
	if (modelId >= modelList.size()) return;
	modelList[modelId].setModel(newModel);
}

void VulkanRender::setModelVisible(int modelId, bool visible)
{
	if (modelId >= 0 && modelId < (int)modelVisible.size())
		modelVisible[modelId] = visible;
}

void VulkanRender::updateLight(LightUbo lightDatay)
{
	lightData = lightDatay;
}

void VulkanRender::draw()
{
	uint32_t imageIndex;
	// Wait for all in-flight frames so the acquire semaphore is not still pending from a previous frame
	vkWaitForFences(mainDevice.logicalDevice, static_cast<uint32_t>(drawFences.size()), drawFences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max());
	VkResult result = vkAcquireNextImageKHR(mainDevice.logicalDevice, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailable[0], VK_NULL_HANDLE, &imageIndex);
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("Failed to acquire swapchain image");
	}
	vkResetFences(mainDevice.logicalDevice, 1, &drawFences[imageIndex]);

	updateUniformBuffers(imageIndex);
	recordCommands(imageIndex);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &imageAvailable[0];  // wait on the semaphore acquire signaled
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &renderFinished[imageIndex];  // per-image: never reuse until this image is presented

	result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, drawFences[imageIndex]);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to submit command buffer");
	}

	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinished[imageIndex];  // present waits on this image's semaphore only
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapChain;
	presentInfo.pImageIndices = &imageIndex;

	result = vkQueuePresentKHR(presentationQueue, &presentInfo);
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("Failed to present");
	}
}

void VulkanRender::cleanup()
{

	vkDeviceWaitIdle(mainDevice.logicalDevice);
	//_aligned_free(modelTransferSpace);

	for (size_t i = 0; i < modelList.size(); i++) {
		modelList[i].destroyMeshModel();
	}

	vkDestroyDescriptorPool(mainDevice.logicalDevice, inputDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, inputSetLayout, nullptr);
	vkDestroyDescriptorPool(mainDevice.logicalDevice, samplerDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, samplerSetLayout, nullptr);
	vkDestroySampler(mainDevice.logicalDevice, textureSampler, nullptr);

	vkDestroySampler(mainDevice.logicalDevice, shadowMapSampler, nullptr);
	vkDestroyDescriptorPool(mainDevice.logicalDevice, shadowMapDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, shadowMapSetLayout, nullptr);
	vkDestroyBuffer(mainDevice.logicalDevice, shadowMapUniformBuffer, nullptr);
	vkFreeMemory(mainDevice.logicalDevice, shadowMapUniformBufferMemory, nullptr);
	vkDestroyPipeline(mainDevice.logicalDevice, shadowMapPipeline, nullptr);
	vkDestroyPipelineLayout(mainDevice.logicalDevice, shadowMapPipelineLayout, nullptr);
	if (gizmoLinePipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(mainDevice.logicalDevice, gizmoLinePipeline, nullptr);
		gizmoLinePipeline = VK_NULL_HANDLE;
	}
	if (gizmoLinePipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(mainDevice.logicalDevice, gizmoLinePipelineLayout, nullptr);
		gizmoLinePipelineLayout = VK_NULL_HANDLE;
	}
	if (gizmoIndexBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(mainDevice.logicalDevice, gizmoIndexBuffer, nullptr);
		vkFreeMemory(mainDevice.logicalDevice, gizmoIndexBufferMemory, nullptr);
		gizmoIndexBuffer = VK_NULL_HANDLE;
	}
	if (gizmoVertexBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(mainDevice.logicalDevice, gizmoVertexBuffer, nullptr);
		vkFreeMemory(mainDevice.logicalDevice, gizmoVertexBufferMemory, nullptr);
		gizmoVertexBuffer = VK_NULL_HANDLE;
	}
	vkDestroyFramebuffer(mainDevice.logicalDevice, shadowMapFramebuffer, nullptr);
	vkDestroyRenderPass(mainDevice.logicalDevice, shadowMapRenderPass, nullptr);
	vkDestroyImageView(mainDevice.logicalDevice, shadowMapImageView, nullptr);
	vkDestroyImage(mainDevice.logicalDevice, shadowMapImage, nullptr);
	vkFreeMemory(mainDevice.logicalDevice, shadowMapMemory, nullptr);

	for (size_t i = 0; i < textureImages.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, textureImageViews[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, textureImages[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, textureImageMemory[i], nullptr);
	}


	for (size_t i = 0; i < swapChainImages.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, depthBufferImageView[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, depthBufferImages[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, depthBufferImageMemory[i], nullptr);
	}

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, colourBufferImageView[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, colourBufferImages[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, colourBufferImageMemory[i], nullptr);
	}



	vkDestroyDescriptorPool(mainDevice.logicalDevice, descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, descriptorSetLayout, nullptr);
	for (size_t i = 0; i < swapChainImages.size(); i++) {
		vkDestroyBuffer(mainDevice.logicalDevice, vpUniformBuffer[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, vpUniformBufferMemory[i], nullptr);
		//vkDestroyBuffer(mainDevice.logicalDevice, modelUniformBuffer[i], nullptr);
		//vkFreeMemory(mainDevice.logicalDevice, modelUniformBufferMemory[i], nullptr);

		vkDestroyBuffer(mainDevice.logicalDevice, lUniformBuffer[i], nullptr); // <<< EKLENECEK
		vkFreeMemory(mainDevice.logicalDevice, lUniformBufferMemory[i], nullptr); // 
	}

	for (size_t i = 0; i < imageAvailable.size(); i++)
		vkDestroySemaphore(mainDevice.logicalDevice, imageAvailable[i], nullptr);
	for (size_t i = 0; i < renderFinished.size(); i++)
		vkDestroySemaphore(mainDevice.logicalDevice, renderFinished[i], nullptr);
	for (size_t i = 0; i < drawFences.size(); i++)
		vkDestroyFence(mainDevice.logicalDevice, drawFences[i], nullptr);

	vkDestroyCommandPool(mainDevice.logicalDevice, graphicsCommandPool, nullptr);
	for (auto framebuffer : swapChainFramebuffers) {
		vkDestroyFramebuffer(mainDevice.logicalDevice, framebuffer, nullptr);
	}
	if (secondPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(mainDevice.logicalDevice, secondPipeline, nullptr);
		secondPipeline = VK_NULL_HANDLE;
	}
	if (secondPipeLineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(mainDevice.logicalDevice, secondPipeLineLayout, nullptr);
		secondPipeLineLayout = VK_NULL_HANDLE;
	}
	if (skyPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(mainDevice.logicalDevice, skyPipeline, nullptr);
		skyPipeline = VK_NULL_HANDLE;
	}
	if (skyPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(mainDevice.logicalDevice, skyPipelineLayout, nullptr);
		skyPipelineLayout = VK_NULL_HANDLE;
	}
	if (starPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(mainDevice.logicalDevice, starPipeline, nullptr);
		starPipeline = VK_NULL_HANDLE;
	}
	if (starPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(mainDevice.logicalDevice, starPipelineLayout, nullptr);
		starPipelineLayout = VK_NULL_HANDLE;
	}
	if (starVertexBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(mainDevice.logicalDevice, starVertexBuffer, nullptr);
		starVertexBuffer = VK_NULL_HANDLE;
	}
	if (starVertexBufferMemory != VK_NULL_HANDLE) {
		vkFreeMemory(mainDevice.logicalDevice, starVertexBufferMemory, nullptr);
		starVertexBufferMemory = VK_NULL_HANDLE;
	}
	if (starIndexBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(mainDevice.logicalDevice, starIndexBuffer, nullptr);
		starIndexBuffer = VK_NULL_HANDLE;
	}
	if (starIndexBufferMemory != VK_NULL_HANDLE) {
		vkFreeMemory(mainDevice.logicalDevice, starIndexBufferMemory, nullptr);
		starIndexBufferMemory = VK_NULL_HANDLE;
	}
	vkDestroyPipeline(mainDevice.logicalDevice, graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(mainDevice.logicalDevice, pipelineLayout, nullptr);
	vkDestroyRenderPass(mainDevice.logicalDevice, renderPass, nullptr);
	for (auto image : swapChainImages) {
		vkDestroyImageView(mainDevice.logicalDevice, image.imageView, nullptr);
	}
	vkDestroySwapchainKHR(mainDevice.logicalDevice, swapChain, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyDevice(mainDevice.logicalDevice, nullptr);
	if (enableValidationLayers) {
		destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	}
	vkDestroyInstance(instance, nullptr);
}

VulkanRender::~VulkanRender()
{
}

void VulkanRender::createInstance()
{
	if (enableValidationLayers && !checkValidationLayerSupport()) {
		throw std::runtime_error("Validation layers requested, but not available!");
	}




	VkApplicationInfo applicationInfo = {};
	applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	applicationInfo.pApplicationName = "Vulkan app";
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pEngineName = "Custom Engine";
	applicationInfo.apiVersion = VK_API_VERSION_1_2;


	//creation information for a Vk instance
	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &applicationInfo;

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
	if (enableValidationLayers) {


		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		populateDebugMessengerCreateInfo(debugCreateInfo);
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;
		createInfo.pNext = nullptr;
	}



	std::vector<const char*> extensions = getRequiredExtensions();

	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();


	VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a vulkan instance");
	}

}

void VulkanRender::createDebugMessenger()
{
	if (!enableValidationLayers) return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
	populateDebugMessengerCreateInfo(createInfo);

	if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("Failed to set up debug messenger!");
	}
}


void VulkanRender::createLogicalDevice()
{

	QueueFamilyIndices indices = getQueueFamilyIndices(mainDevice.physicalDevice);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<int> queueFamilyIndices = { indices.graphicsFamily,indices.presentationFamily };


	for (int queueFamilyIndex : queueFamilyIndices) {
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = 1;
		float priority = 1.0f;
		queueCreateInfo.pQueuePriorities = &priority;

		queueCreateInfos.push_back(queueCreateInfo);
	}


	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();



	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();


	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.depthClamp = VK_TRUE;  // allow shadow pass to clamp depth instead of clipping
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;


	VkResult result = vkCreateDevice(mainDevice.physicalDevice, &deviceCreateInfo, nullptr, &mainDevice.logicalDevice);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create a Logical Device!");
	}


	vkGetDeviceQueue(mainDevice.logicalDevice, indices.graphicsFamily, 0, &graphicsQueue);
	vkGetDeviceQueue(mainDevice.logicalDevice, indices.presentationFamily, 0, &presentationQueue);
}

void VulkanRender::createSurface()
{
	VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create a surface!");
	}
}

void VulkanRender::createSwapChain()
{
	SwapChainDetails swapChainDetails = getSwapChainDetails(mainDevice.physicalDevice);

	VkSurfaceFormatKHR surfaceFormat = chooseBestSurfaceFromat(swapChainDetails.formats);
	VkPresentModeKHR presentMode = chooseBestPresentationMode(swapChainDetails.presentModes);
	VkExtent2D extent = chooseSwapExtent(swapChainDetails.surfaceCapabilities);

	uint32_t imageCount = swapChainDetails.surfaceCapabilities.minImageCount + 1;

	if (swapChainDetails.surfaceCapabilities.maxImageCount > 0 && swapChainDetails.surfaceCapabilities.maxImageCount < imageCount) {
		imageCount = swapChainDetails.surfaceCapabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
	swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapChainCreateInfo.imageFormat = surfaceFormat.format;
	swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
	swapChainCreateInfo.presentMode = presentMode;
	swapChainCreateInfo.imageExtent = extent;
	swapChainCreateInfo.minImageCount = imageCount;
	swapChainCreateInfo.imageArrayLayers = 1;
	swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapChainCreateInfo.preTransform = swapChainDetails.surfaceCapabilities.currentTransform;
	swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapChainCreateInfo.clipped = VK_TRUE;
	swapChainCreateInfo.surface = surface;

	QueueFamilyIndices indices = getQueueFamilyIndices(mainDevice.physicalDevice);

	if (indices.graphicsFamily != indices.presentationFamily) {

		uint32_t queueFamilyIndices[] = {
			(uint32_t)indices.graphicsFamily,
			(uint32_t)indices.presentationFamily
		};
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapChainCreateInfo.queueFamilyIndexCount = 2;
		swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		swapChainCreateInfo.queueFamilyIndexCount = 0;
		swapChainCreateInfo.pQueueFamilyIndices = nullptr;
	}

	swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;


	VkResult result = vkCreateSwapchainKHR(mainDevice.logicalDevice, &swapChainCreateInfo, nullptr, &swapChain);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create a swapchain!");
	}

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;



	uint32_t swapChainImageCount = 0;
	vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapChain, &swapChainImageCount, nullptr);


	std::vector<VkImage> images(swapChainImageCount);
	vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapChain, &swapChainImageCount, images.data());


	for (VkImage image : images) {
		SwapchainImage swapChainImage = {};
		swapChainImage.image = image;
		swapChainImage.imageView = createImageView(image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);

		swapChainImages.push_back(swapChainImage);

	}
}

void VulkanRender::createRenderPass()
{
	std::array<VkSubpassDescription, 2> subpasses{};

	// depth attachment (colour is swapchain only; no middle colour buffer)
	VkAttachmentDescription depthAttachment = {};
	depthAttachment.format = chooseSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// Single subpass: draw directly to swapchain (0) and depth (1). Second pass disabled.
	VkAttachmentReference colorAttachmentReference = {};
	colorAttachmentReference.attachment = 0;
	colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	VkAttachmentReference depthAttachmentReference = {};
	depthAttachmentReference.attachment = 1;
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[0].colorAttachmentCount = 1;
	subpasses[0].pColorAttachments = &colorAttachmentReference;
	subpasses[0].pDepthStencilAttachment = &depthAttachmentReference;



	VkAttachmentDescription swapChainColourAttachment = {};
	swapChainColourAttachment.format = swapChainImageFormat;
	swapChainColourAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	swapChainColourAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	swapChainColourAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	swapChainColourAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	swapChainColourAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;


	swapChainColourAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	swapChainColourAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;






	VkAttachmentReference swapchainColourAttachmentReference = {};
	swapchainColourAttachmentReference.attachment = 0;
	swapchainColourAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


	std::array<VkAttachmentReference, 2> inputReferences;
	inputReferences[0].attachment = 1;
	inputReferences[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	inputReferences[1].attachment = 2;
	inputReferences[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;


	subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[1].colorAttachmentCount = 1;
	subpasses[1].pColorAttachments = &swapchainColourAttachmentReference;
	subpasses[1].inputAttachmentCount = static_cast<uint32_t>(inputReferences.size());
	subpasses[1].pInputAttachments = inputReferences.data();







	std::array<VkSubpassDependency, 2> subpassDependencies;

	subpassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	subpassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	subpassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;

	subpassDependencies[0].dstSubpass = 0;
	subpassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	subpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	subpassDependencies[0].dependencyFlags = 0;

	subpassDependencies[1].srcSubpass = 0;
	subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	subpassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	subpassDependencies[1].dependencyFlags = 0;


	std::array<VkAttachmentDescription, 2> renderPassAttachments = { swapChainColourAttachment, depthAttachment };

	VkRenderPassCreateInfo renderPassCreateInfo = {};
	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(renderPassAttachments.size());
	renderPassCreateInfo.pAttachments = renderPassAttachments.data();
	renderPassCreateInfo.subpassCount = 1;
	renderPassCreateInfo.pSubpasses = subpasses.data();
	renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
	renderPassCreateInfo.pDependencies = subpassDependencies.data();


	VkResult result = vkCreateRenderPass(mainDevice.logicalDevice, &renderPassCreateInfo, nullptr, &renderPass);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a render pass!");
	}
}

void VulkanRender::createDescriptorSetLayout()
{


	VkDescriptorSetLayoutBinding vpLayoutBinding = {};
	vpLayoutBinding.binding = 0;
	vpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vpLayoutBinding.descriptorCount = 1;
	vpLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	vpLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding lLayoutBinding = {};
	lLayoutBinding.binding = 1;
	lLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	lLayoutBinding.descriptorCount = 1;
	lLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	lLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding shadowSamplerBinding = {};
	shadowSamplerBinding.binding = 2;
	shadowSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	shadowSamplerBinding.descriptorCount = 1;
	shadowSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	shadowSamplerBinding.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> layoutBindings = { vpLayoutBinding, lLayoutBinding, shadowSamplerBinding };

	VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutCreateInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
	layoutCreateInfo.pBindings = layoutBindings.data();


	VkResult result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &layoutCreateInfo, nullptr, &descriptorSetLayout);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create desriptor set layout");
	}

	VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
	samplerLayoutBinding.binding = 0;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	samplerLayoutBinding.pImmutableSamplers = nullptr;


	VkDescriptorSetLayoutCreateInfo textureLayoutCreateInfo = {};
	textureLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	textureLayoutCreateInfo.bindingCount = 1;
	textureLayoutCreateInfo.pBindings = &samplerLayoutBinding;

	result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &textureLayoutCreateInfo, nullptr, &samplerSetLayout);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create desriptor set layout");
	}

	VkDescriptorSetLayoutBinding colorInputLayoutBinding = {};
	colorInputLayoutBinding.binding = 0;
	colorInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	colorInputLayoutBinding.descriptorCount = 1;
	colorInputLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutBinding depthInputLayoutBinding = {};
	depthInputLayoutBinding.binding = 1;
	depthInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	depthInputLayoutBinding.descriptorCount = 1;
	depthInputLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	std::vector<VkDescriptorSetLayoutBinding> inputBindings = { colorInputLayoutBinding,depthInputLayoutBinding };

	VkDescriptorSetLayoutCreateInfo inputLayoutCreateInfo = {};
	inputLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	inputLayoutCreateInfo.bindingCount = static_cast<uint32_t>(inputBindings.size());
	inputLayoutCreateInfo.pBindings = inputBindings.data();


	result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &inputLayoutCreateInfo, nullptr, &inputSetLayout);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create desriptor set layout");
	}


}

void VulkanRender::createPushConstantRange()
{
	pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(Model);
}

void VulkanRender::createGraphicsPipeLine()
{
	auto vertexShaderCode = readFile("Shaders/vert.spv");
	auto fragmentShaderCode = readFile("Shaders/frag.spv");

	VkShaderModule vertexShaderModule = createShaderModule(vertexShaderCode);
	VkShaderModule fragmentShaderModule = createShaderModule(fragmentShaderCode);


	VkPipelineShaderStageCreateInfo vertexShaderCreateInfo = {};
	vertexShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertexShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertexShaderCreateInfo.module = vertexShaderModule;
	vertexShaderCreateInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragmentShaderCreateInfo = {};
	fragmentShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragmentShaderCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragmentShaderCreateInfo.module = fragmentShaderModule;
	fragmentShaderCreateInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertexShaderCreateInfo,fragmentShaderCreateInfo };

	VkVertexInputBindingDescription bindingDescription = {};
	bindingDescription.binding = 0;
	bindingDescription.stride = sizeof(Vertex);
	bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	std::array<VkVertexInputAttributeDescription, 4> attributeDesritions;

	attributeDesritions[0].binding = 0;
	attributeDesritions[0].location = 0;
	attributeDesritions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDesritions[0].offset = offsetof(Vertex, pos);

	attributeDesritions[1].binding = 0;
	attributeDesritions[1].location = 1;
	attributeDesritions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDesritions[1].offset = offsetof(Vertex, color);

	attributeDesritions[2].binding = 0;
	attributeDesritions[2].location = 2;
	attributeDesritions[2].format = VK_FORMAT_R32G32_SFLOAT;
	attributeDesritions[2].offset = offsetof(Vertex, tex);

	attributeDesritions[3].binding = 0;
	attributeDesritions[3].location = 3;
	attributeDesritions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDesritions[3].offset = offsetof(Vertex, normal);



	VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
	vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
	vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDesritions.size());
	vertexInputCreateInfo.pVertexAttributeDescriptions = &attributeDesritions[0];


	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;


	VkViewport viewport = {};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)swapChainExtent.width;
	viewport.height = (float)swapChainExtent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;


	VkRect2D scissor = {};
	scissor.offset = { 0,0 };
	scissor.extent = swapChainExtent;

	VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {};
	viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateCreateInfo.viewportCount = 1;
	viewportStateCreateInfo.pViewports = &viewport;
	viewportStateCreateInfo.scissorCount = 1;
	viewportStateCreateInfo.pScissors = &scissor;


	/*
	std::vector<VkDynamicState> dynamicStateEnables;
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);

	VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {};
	dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());               //// e?er pencere resize edilmeli ise buras? kullan?lmal?
	dynamicStateCreateInfo.pDynamicStates = dynamicStateEnables.data();
	*/



	VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo = {};
	rasterizerCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizerCreateInfo.depthClampEnable = VK_FALSE;
	rasterizerCreateInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterizerCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizerCreateInfo.lineWidth = 1.0f;
	rasterizerCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizerCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizerCreateInfo.depthBiasEnable = VK_FALSE;



	VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo = {};
	multisamplingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisamplingCreateInfo.sampleShadingEnable = VK_FALSE;
	multisamplingCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState colourState = {};
	colourState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colourState.blendEnable = VK_TRUE;

	colourState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colourState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colourState.colorBlendOp = VK_BLEND_OP_ADD;

	colourState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colourState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colourState.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colourBlendingCreateInfo = {};
	colourBlendingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colourBlendingCreateInfo.logicOpEnable = VK_FALSE;
	colourBlendingCreateInfo.attachmentCount = 1;
	colourBlendingCreateInfo.pAttachments = &colourState;

	std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = { descriptorSetLayout, samplerSetLayout };

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
	pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

	VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create a pipeline");
	}


	VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo = {};
	depthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilCreateInfo.depthTestEnable = VK_TRUE;
	depthStencilCreateInfo.depthWriteEnable = VK_TRUE;
	depthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE;
	depthStencilCreateInfo.stencilTestEnable = VK_FALSE;




	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.stageCount = 2;
	pipelineCreateInfo.pStages = shaderStages;
	pipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
	pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
	pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
	pipelineCreateInfo.pDynamicState = nullptr;
	pipelineCreateInfo.pRasterizationState = &rasterizerCreateInfo;
	pipelineCreateInfo.pMultisampleState = &multisamplingCreateInfo;
	pipelineCreateInfo.pColorBlendState = &colourBlendingCreateInfo;
	pipelineCreateInfo.pDepthStencilState = &depthStencilCreateInfo;
	pipelineCreateInfo.layout = pipelineLayout;
	pipelineCreateInfo.renderPass = renderPass;
	pipelineCreateInfo.subpass = 0;


	pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	pipelineCreateInfo.basePipelineIndex = -1;


	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline);


	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create a graphics pipeline");
	}


	vkDestroyShaderModule(mainDevice.logicalDevice, fragmentShaderModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, vertexShaderModule, nullptr);

	/* Second pipeline disabled: render pass has only 1 subpass (0). Creating for subpass 1 would fail. */
}
void VulkanRender::createColourBufferImage()
{
	colourBufferImages.resize(swapChainImages.size());
	colourBufferImageMemory.resize(swapChainImages.size());
	colourBufferImageView.resize(swapChainImages.size());

	VkFormat colourFormat = chooseSupportedFormat(
		{ VK_FORMAT_R8G8B8A8_UNORM },
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	);

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		colourBufferImages[i] = createImage(swapChainExtent.width, swapChainExtent.height, colourFormat, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &colourBufferImageMemory[i]);

		colourBufferImageView[i] = createImageView(colourBufferImages[i], colourFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}
}

void VulkanRender::createDepthBuffer()
{
	depthBufferImages.resize(swapChainImages.size());
	depthBufferImageMemory.resize(swapChainImages.size());
	depthBufferImageView.resize(swapChainImages.size());

	depthFormat = chooseSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		depthBufferImages[i] = createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &depthBufferImageMemory[i]);

		depthBufferImageView[i] = createImageView(depthBufferImages[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
	}

	createShadowMapImage();
	createShadowMapRenderPass();
	createShadowMapFramebuffer();
	createShadowMapDescriptors();  // creates set layout, UBO, descriptor set, and shadow sampler
	createShadowMapPipeline();
}

void VulkanRender::createFramebuffers()
{
	swapChainFramebuffers.resize(swapChainImages.size());
	for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
		std::array<VkImageView, 2> attachments = {
			swapChainImages[i].imageView,
			depthBufferImageView[i]
		};
		VkFramebufferCreateInfo framebufferCreateInfo = {};
		framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferCreateInfo.renderPass = renderPass;
		framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferCreateInfo.pAttachments = attachments.data();         /////////////// burada kald?n
		framebufferCreateInfo.width = swapChainExtent.width;
		framebufferCreateInfo.height = swapChainExtent.height;
		framebufferCreateInfo.layers = 1;


		VkResult result = vkCreateFramebuffer(mainDevice.logicalDevice, &framebufferCreateInfo, nullptr, &swapChainFramebuffers[i]);

		if (result != VK_SUCCESS) {
			throw std::runtime_error("Failed to create a framebuffer");
		}
	}
}

void VulkanRender::createShadowMapImage()
{
	// Prefer depth-only format so barriers can use DEPTH_BIT only (avoids D32_SFLOAT_S8_UINT stencil aspect)
	shadowMapDepthFormat = chooseSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);
	shadowMapImage = createImage(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, shadowMapDepthFormat, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &shadowMapMemory);
	shadowMapImageView = createImageView(shadowMapImage, shadowMapDepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void VulkanRender::createShadowMapRenderPass()
{
	VkAttachmentDescription depthAttachment = {};
	depthAttachment.format = shadowMapDepthFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

	VkAttachmentReference depthRef = {};
	depthRef.attachment = 0;
	depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 0;
	subpass.pDepthStencilAttachment = &depthRef;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo rpInfo = {};
	rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	rpInfo.attachmentCount = 1;
	rpInfo.pAttachments = &depthAttachment;
	rpInfo.subpassCount = 1;
	rpInfo.pSubpasses = &subpass;
	rpInfo.dependencyCount = 1;
	rpInfo.pDependencies = &dependency;

	VkResult result = vkCreateRenderPass(mainDevice.logicalDevice, &rpInfo, nullptr, &shadowMapRenderPass);
	if (result != VK_SUCCESS)
		throw std::runtime_error("Failed to create shadow map render pass");
}

void VulkanRender::createShadowMapFramebuffer()
{
	VkFramebufferCreateInfo fbInfo = {};
	fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fbInfo.renderPass = shadowMapRenderPass;
	fbInfo.attachmentCount = 1;
	fbInfo.pAttachments = &shadowMapImageView;
	fbInfo.width = SHADOW_MAP_SIZE;
	fbInfo.height = SHADOW_MAP_SIZE;
	fbInfo.layers = 1;

	VkResult result = vkCreateFramebuffer(mainDevice.logicalDevice, &fbInfo, nullptr, &shadowMapFramebuffer);
	if (result != VK_SUCCESS)
		throw std::runtime_error("Failed to create shadow map framebuffer");
}

void VulkanRender::createShadowMapPipeline()
{
	auto vertCode = readFile("Shaders/shadow_vert.spv");
	auto fragCode = readFile("Shaders/shadow_frag.spv");
	VkShaderModule vertModule = createShaderModule(vertCode);
	VkShaderModule fragModule = createShaderModule(fragCode);

	VkPipelineShaderStageCreateInfo vertStage = {};
	vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertStage.module = vertModule;
	vertStage.pName = "main";
	VkPipelineShaderStageCreateInfo fragStage = {};
	fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragStage.module = fragModule;
	fragStage.pName = "main";

	VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

	VkVertexInputBindingDescription binding = {};
	binding.binding = 0;
	binding.stride = sizeof(Vertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	// VkVertexInputAttributeDescription order is (location, binding, format, offset)
	std::array<VkVertexInputAttributeDescription, 4> attrs = {};
	attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
	attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };
	attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, tex) };
	attrs[3] = { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };

	VkPipelineVertexInputStateCreateInfo vertexInput = {};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
	vertexInput.pVertexAttributeDescriptions = attrs.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(SHADOW_MAP_SIZE);
	viewport.height = static_cast<float>(SHADOW_MAP_SIZE);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor = { { 0, 0 }, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE } };
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo raster = {};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.depthClampEnable = VK_TRUE;   // clamp depth so geometry outside [near,far] still writes to shadow map
	raster.rasterizerDiscardEnable = VK_FALSE;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.lineWidth = 1.0f;
	/* No culling: draw all faces so every object writes depth (avoids empty shadow map / single black rect) */
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	// No depth bias here: we only bias in the fragment shader. Double-bias (here + shader) hid shadows.
	raster.depthBiasEnable = VK_FALSE;
	raster.depthBiasConstantFactor = 0.0f;
	raster.depthBiasSlopeFactor = 0.0f;
	raster.depthBiasClamp = 0.0f;

	VkPipelineMultisampleStateCreateInfo multisample = {};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.sampleShadingEnable = VK_FALSE;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	VkPipelineLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts = &shadowMapSetLayout;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &pushConstantRange;
	vkCreatePipelineLayout(mainDevice.logicalDevice, &layoutInfo, nullptr, &shadowMapPipelineLayout);

	VkGraphicsPipelineCreateInfo pipeInfo = {};
	pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeInfo.stageCount = 2;
	pipeInfo.pStages = stages;
	pipeInfo.pVertexInputState = &vertexInput;
	pipeInfo.pInputAssemblyState = &inputAssembly;
	pipeInfo.pViewportState = &viewportState;
	pipeInfo.pRasterizationState = &raster;
	pipeInfo.pMultisampleState = &multisample;
	pipeInfo.pDepthStencilState = &depthStencil;
	pipeInfo.layout = shadowMapPipelineLayout;
	pipeInfo.renderPass = shadowMapRenderPass;
	pipeInfo.subpass = 0;
	VkResult result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &shadowMapPipeline);
	vkDestroyShaderModule(mainDevice.logicalDevice, vertModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, fragModule, nullptr);
	if (result != VK_SUCCESS)
		throw std::runtime_error("Failed to create shadow map pipeline");
}

struct GizmoVertex { glm::vec3 pos; glm::vec3 color; };

void VulkanRender::createGizmoPipeline()
{
	auto vertCode = readFile("Shaders/line_vert.spv");
	auto fragCode = readFile("Shaders/line_frag.spv");
	VkShaderModule vertModule = createShaderModule(vertCode);
	VkShaderModule fragModule = createShaderModule(fragCode);

	VkPipelineShaderStageCreateInfo vertStage = {};
	vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertStage.module = vertModule;
	vertStage.pName = "main";
	VkPipelineShaderStageCreateInfo fragStage = {};
	fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragStage.module = fragModule;
	fragStage.pName = "main";
	VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

	VkVertexInputBindingDescription binding = {};
	binding.binding = 0;
	binding.stride = sizeof(GizmoVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	VkVertexInputAttributeDescription attrs[2] = {};
	attrs[0].binding = 0; attrs[0].location = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = offsetof(GizmoVertex, pos);
	attrs[1].binding = 0; attrs[1].location = 1; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(GizmoVertex, color);

	VkPipelineVertexInputStateCreateInfo vertexInput = {};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 2;
	vertexInput.pVertexAttributeDescriptions = attrs;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};
	viewport.x = 0.0f; viewport.y = 0.0f;
	viewport.width = static_cast<float>(swapChainExtent.width);
	viewport.height = static_cast<float>(swapChainExtent.height);
	viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
	VkRect2D scissor = { { 0, 0 }, swapChainExtent };
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1; viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo raster = {};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.depthClampEnable = VK_FALSE;
	raster.rasterizerDiscardEnable = VK_FALSE;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.lineWidth = 1.0f;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisample = {};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.sampleShadingEnable = VK_FALSE;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState blendAtt = {};
	blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAtt.blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo blendState = {};
	blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blendState.logicOpEnable = VK_FALSE;
	blendState.attachmentCount = 1;
	blendState.pAttachments = &blendAtt;

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_FALSE;  // don't write depth so scene isn't occluded
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	VkPushConstantRange gizmoPush = {};
	gizmoPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	gizmoPush.offset = 0;
	gizmoPush.size = sizeof(glm::mat4);

	VkPipelineLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 0;
	layoutInfo.pSetLayouts = nullptr;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &gizmoPush;
	VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice, &layoutInfo, nullptr, &gizmoLinePipelineLayout);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create gizmo pipeline layout");

	VkGraphicsPipelineCreateInfo pipeInfo = {};
	pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeInfo.stageCount = 2;
	pipeInfo.pStages = stages;
	pipeInfo.pVertexInputState = &vertexInput;
	pipeInfo.pInputAssemblyState = &inputAssembly;
	pipeInfo.pViewportState = &viewportState;
	pipeInfo.pRasterizationState = &raster;
	pipeInfo.pMultisampleState = &multisample;
	pipeInfo.pColorBlendState = &blendState;
	pipeInfo.pDepthStencilState = &depthStencil;
	pipeInfo.pDynamicState = nullptr;
	pipeInfo.layout = gizmoLinePipelineLayout;
	pipeInfo.renderPass = renderPass;
	pipeInfo.subpass = 0;
	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &gizmoLinePipeline);
	vkDestroyShaderModule(mainDevice.logicalDevice, vertModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, fragModule, nullptr);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create gizmo line pipeline");

	const size_t gizmoBufferSize = 12 * sizeof(GizmoVertex);  // 4 verts per axis (kalin dortgen)
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, gizmoBufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&gizmoVertexBuffer, &gizmoVertexBufferMemory);
	// Indices: 2 ucgensel per axis (0,1,2, 0,2,3), (4,5,6, 4,6,7), (8,9,10, 8,10,11)
	const uint32_t gizmoIndices[] = { 0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11 };
	const size_t gizmoIndexSize = 18 * sizeof(uint32_t);
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, gizmoIndexSize,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&gizmoIndexBuffer, &gizmoIndexBufferMemory);
	void* idxData;
	vkMapMemory(mainDevice.logicalDevice, gizmoIndexBufferMemory, 0, gizmoIndexSize, 0, &idxData);
	memcpy(idxData, gizmoIndices, gizmoIndexSize);
	vkUnmapMemory(mainDevice.logicalDevice, gizmoIndexBufferMemory);
}

void VulkanRender::createSkyPipeline()
{
	auto vertCode = readFile("Shaders/sky_vert.spv");
	auto fragCode = readFile("Shaders/sky_frag.spv");
	VkShaderModule vertModule = createShaderModule(vertCode);
	VkShaderModule fragModule = createShaderModule(fragCode);

	VkPipelineShaderStageCreateInfo vertStage = {};
	vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertStage.module = vertModule;
	vertStage.pName = "main";
	VkPipelineShaderStageCreateInfo fragStage = {};
	fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragStage.module = fragModule;
	fragStage.pName = "main";
	VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

	// No vertex buffer: fullscreen triangle from vertex index
	VkPipelineVertexInputStateCreateInfo vertexInput = {};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 0;
	vertexInput.vertexAttributeDescriptionCount = 0;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};
	viewport.x = 0.0f; viewport.y = 0.0f;
	viewport.width = static_cast<float>(swapChainExtent.width);
	viewport.height = static_cast<float>(swapChainExtent.height);
	viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
	VkRect2D scissor = { { 0, 0 }, swapChainExtent };
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1; viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo raster = {};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.depthClampEnable = VK_FALSE;
	raster.rasterizerDiscardEnable = VK_FALSE;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.lineWidth = 1.0f;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisample = {};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.sampleShadingEnable = VK_FALSE;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState blendAtt = {};
	blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAtt.blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo blendState = {};
	blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blendState.logicOpEnable = VK_FALSE;
	blendState.attachmentCount = 1;
	blendState.pAttachments = &blendAtt;

	// Sky at far plane; depth write off so scene overwrites
	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	// Push: invViewProj (64), cameraPos+pad (16), skyTop+pad (16), skyBottom+pad (16) = 112
	struct SkyPush {
		glm::mat4 invViewProj;
		glm::vec3 cameraPos; float pad0;
		glm::vec3 skyTop;    float pad1;
		glm::vec3 skyBottom; float pad2;
	};
	VkPushConstantRange skyPush = {};
	skyPush.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	skyPush.offset = 0;
	skyPush.size = sizeof(SkyPush);

	VkPipelineLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 0;
	layoutInfo.pSetLayouts = nullptr;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &skyPush;
	VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice, &layoutInfo, nullptr, &skyPipelineLayout);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create sky pipeline layout");

	VkGraphicsPipelineCreateInfo pipeInfo = {};
	pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeInfo.stageCount = 2;
	pipeInfo.pStages = stages;
	pipeInfo.pVertexInputState = &vertexInput;
	pipeInfo.pInputAssemblyState = &inputAssembly;
	pipeInfo.pViewportState = &viewportState;
	pipeInfo.pRasterizationState = &raster;
	pipeInfo.pMultisampleState = &multisample;
	pipeInfo.pColorBlendState = &blendState;
	pipeInfo.pDepthStencilState = &depthStencil;
	pipeInfo.pDynamicState = nullptr;
	pipeInfo.layout = skyPipelineLayout;
	pipeInfo.renderPass = renderPass;
	pipeInfo.subpass = 0;
	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &skyPipeline);
	vkDestroyShaderModule(mainDevice.logicalDevice, vertModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, fragModule, nullptr);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create sky pipeline");
}

struct StarVertex { glm::vec3 pos; glm::vec3 normal; };

void VulkanRender::createStarSphere() {
	const int segments = 48;
	const int rings = 32;
	std::vector<StarVertex> verts;
	std::vector<uint32_t> indices;
	for (int r = 0; r <= rings; r++) {
		float v = (float)r / (float)rings;
		float phi = v * 3.14159265f;
		float y = cos(phi);
		float ringRadius = sin(phi);
		for (int s = 0; s <= segments; s++) {
			float u = (float)s / (float)segments;
			float theta = u * 2.0f * 3.14159265f;
			float x = ringRadius * cos(theta);
			float z = ringRadius * sin(theta);
			glm::vec3 p(x, y, z);
			verts.push_back({ p, glm::normalize(p) });
		}
	}
	for (int r = 0; r < rings; r++) {
		for (int s = 0; s < segments; s++) {
			int i0 = r * (segments + 1) + s;
			int i1 = i0 + 1;
			int i2 = i0 + (segments + 1);
			int i3 = i2 + 1;
			indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
			indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
		}
	}
	starIndexCount = static_cast<uint32_t>(indices.size());
	VkDeviceSize vbSize = verts.size() * sizeof(StarVertex);
	VkDeviceSize ibSize = indices.size() * sizeof(uint32_t);
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, vbSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&starVertexBuffer, &starVertexBufferMemory);
	void* data;
	vkMapMemory(mainDevice.logicalDevice, starVertexBufferMemory, 0, vbSize, 0, &data);
	memcpy(data, verts.data(), vbSize);
	vkUnmapMemory(mainDevice.logicalDevice, starVertexBufferMemory);
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, ibSize,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&starIndexBuffer, &starIndexBufferMemory);
	vkMapMemory(mainDevice.logicalDevice, starIndexBufferMemory, 0, ibSize, 0, &data);
	memcpy(data, indices.data(), ibSize);
	vkUnmapMemory(mainDevice.logicalDevice, starIndexBufferMemory);
}

void VulkanRender::createStarPipeline() {
	auto vertCode = readFile("Shaders/star_vert.spv");
	auto fragCode = readFile("Shaders/star_frag.spv");
	VkShaderModule vertModule = createShaderModule(vertCode);
	VkShaderModule fragModule = createShaderModule(fragCode);
	VkPipelineShaderStageCreateInfo vertStage = {};
	vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertStage.module = vertModule;
	vertStage.pName = "main";
	VkPipelineShaderStageCreateInfo fragStage = {};
	fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragStage.module = fragModule;
	fragStage.pName = "main";
	VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

	VkVertexInputBindingDescription binding = {};
	binding.binding = 0;
	binding.stride = sizeof(StarVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	VkVertexInputAttributeDescription attrs[2] = {};
	attrs[0].binding = 0; attrs[0].location = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0;
	attrs[1].binding = 0; attrs[1].location = 1; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = 12;
	VkPipelineVertexInputStateCreateInfo vertexInput = {};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 2;
	vertexInput.pVertexAttributeDescriptions = attrs;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};
	viewport.x = 0.0f; viewport.y = 0.0f;
	viewport.width = static_cast<float>(swapChainExtent.width);
	viewport.height = static_cast<float>(swapChainExtent.height);
	viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
	VkRect2D scissor = { { 0, 0 }, swapChainExtent };
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1; viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo raster = {};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.depthClampEnable = VK_FALSE;
	raster.rasterizerDiscardEnable = VK_FALSE;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.lineWidth = 1.0f;
	raster.cullMode = VK_CULL_MODE_BACK_BIT;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisample = {};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.sampleShadingEnable = VK_FALSE;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState blendAtt = {};
	blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAtt.blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo blendState = {};
	blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blendState.logicOpEnable = VK_FALSE;
	blendState.attachmentCount = 1;
	blendState.pAttachments = &blendAtt;

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	VkPushConstantRange starPush = {};
	starPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	starPush.offset = 0;
	starPush.size = sizeof(StarParams);

	VkPipelineLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts = &descriptorSetLayout;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &starPush;
	VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice, &layoutInfo, nullptr, &starPipelineLayout);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create star pipeline layout");

	VkGraphicsPipelineCreateInfo pipeInfo = {};
	pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeInfo.stageCount = 2;
	pipeInfo.pStages = stages;
	pipeInfo.pVertexInputState = &vertexInput;
	pipeInfo.pInputAssemblyState = &inputAssembly;
	pipeInfo.pViewportState = &viewportState;
	pipeInfo.pRasterizationState = &raster;
	pipeInfo.pMultisampleState = &multisample;
	pipeInfo.pColorBlendState = &blendState;
	pipeInfo.pDepthStencilState = &depthStencil;
	pipeInfo.pDynamicState = nullptr;
	pipeInfo.layout = starPipelineLayout;
	pipeInfo.renderPass = renderPass;
	pipeInfo.subpass = 0;
	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &starPipeline);
	vkDestroyShaderModule(mainDevice.logicalDevice, vertModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, fragModule, nullptr);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create star pipeline");
}

void VulkanRender::setStarParams(const glm::mat4& model, float timeSec, float temperatureKelvin) {
	starParams.model = model;
	starParams.time = timeSec;
	starParams.temperature = temperatureKelvin;
}

void VulkanRender::updateGizmoBuffer(glm::vec3 worldPos, float objectRadius)
{
	gizmoWorldPos = worldPos;
	gizmoObjectRadius = (objectRadius > 0.0f) ? objectRadius : 1.0f;
	gizmoBaseOffset = gizmoObjectRadius * 0.9f;
	gizmoArrowLength = gizmoObjectRadius * 1.2f;
	const float hr = GIZMO_LINE_THICKNESS * 0.5f;
	const glm::vec3 red(1.0f, 0.15f, 0.15f), green(0.15f, 1.0f, 0.15f), blue(0.2f, 0.2f, 1.0f);
	const glm::vec3 highlight(1.0f, 1.0f, 0.9f);
	glm::vec3 cX = (gizmoHighlightAxis == 1) ? highlight : red;
	glm::vec3 cY = (gizmoHighlightAxis == 2) ? highlight : green;
	glm::vec3 cZ = (gizmoHighlightAxis == 3) ? highlight : blue;
	const glm::vec3 axisX(1, 0, 0), axisY(0, 1, 0), axisZ(0, 0, 1);
	auto quad = [&](const glm::vec3& axisDir, const glm::vec3& right, const glm::vec3& col, GizmoVertex* out) {
		glm::vec3 base = worldPos + axisDir * gizmoBaseOffset;
		glm::vec3 tip = base + axisDir * gizmoArrowLength;
		out[0] = { base - hr * right, col };
		out[1] = { base + hr * right, col };
		out[2] = { tip + hr * right, col };
		out[3] = { tip - hr * right, col };
	};
	GizmoVertex verts[12];
	quad(axisX, glm::vec3(0, 1, 0), cX, &verts[0]);
	quad(axisY, glm::vec3(1, 0, 0), cY, &verts[4]);
	quad(axisZ, glm::vec3(1, 0, 0), cZ, &verts[8]);
	void* data;
	vkMapMemory(mainDevice.logicalDevice, gizmoVertexBufferMemory, 0, 12 * sizeof(GizmoVertex), 0, &data);
	memcpy(data, verts, sizeof(verts));
	vkUnmapMemory(mainDevice.logicalDevice, gizmoVertexBufferMemory);
}

void VulkanRender::getPickRay(float mouseX, float mouseY, float screenW, float screenH, glm::vec3& outOrigin, glm::vec3& outDirection) const {
	float ndcX = (mouseX / screenW) * 2.0f - 1.0f;
	float ndcY = (mouseY / screenH) * 2.0f - 1.0f;  // Vulkan: NDC y=-1 = viewport top, y=+1 = bottom
	glm::mat4 vp = uboViewProjection.projection * cam;  // guncel kamera (uboViewProjection draw'da guncellenir)
	glm::mat4 invVP = glm::inverse(vp);
	glm::vec4 clipNear = invVP * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
	glm::vec4 clipFar  = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
	outOrigin = glm::vec3(clipNear) / clipNear.w;
	glm::vec3 farPt = glm::vec3(clipFar) / clipFar.w;
	outDirection = glm::normalize(farPt - outOrigin);
}

glm::vec3 VulkanRender::getWorldPosOnNearPlane(float mouseX, float mouseY, float screenW, float screenH) const {
	glm::vec3 origin, direction;
	getPickRay(mouseX, mouseY, screenW, screenH, origin, direction);
	return origin;
}

glm::vec3 VulkanRender::getWorldPosOnPlane(float mouseX, float mouseY, float screenW, float screenH, const glm::vec3& planeOrigin, const glm::vec3& planeNormal) const {
	glm::vec3 origin, direction;
	getPickRay(mouseX, mouseY, screenW, screenH, origin, direction);
	float denom = glm::dot(planeNormal, direction);
	const float eps = 1e-5f;
	if (std::abs(denom) < eps)
		return planeOrigin;
	float t = glm::dot(planeOrigin - origin, planeNormal) / denom;
	return origin + t * direction;
}

glm::vec3 VulkanRender::getCameraForward() const {
	// View matrix: camera looks along -Z; third row is world -Z axis
	glm::vec3 forward(-cam[0][2], -cam[1][2], -cam[2][2]);
	float len = glm::length(forward);
	return len > 1e-6f ? forward / len : glm::vec3(0.0f, 0.0f, -1.0f);
}

int VulkanRender::pickGizmoAxis(float mouseX, float mouseY, float screenW, float screenH, const glm::vec3& gizmoWorldPos) const
{
	return pickGizmoAxis(mouseX, mouseY, screenW, screenH, gizmoWorldPos, cam, uboViewProjection.projection);
}

int VulkanRender::pickGizmoAxis(float mouseX, float mouseY, float screenW, float screenH, const glm::vec3& gizmoWorldPos, const glm::mat4& view, const glm::mat4& proj) const
{
	float ndcX = (mouseX / screenW) * 2.0f - 1.0f;
	float ndcY = (mouseY / screenH) * 2.0f - 1.0f;  // Vulkan: NDC y=-1 = viewport top, y=+1 = bottom
	glm::mat4 vp = proj * view;
	glm::mat4 invVP = glm::inverse(vp);
	glm::vec4 clipNear = invVP * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
	glm::vec4 clipFar  = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
	glm::vec3 rayOrig = glm::vec3(clipNear) / clipNear.w;
	glm::vec3 rayDir = glm::normalize(glm::vec3(clipFar) / clipFar.w - rayOrig);

	glm::vec3 camPos = glm::vec3(glm::inverse(view)[3]);
	float dist = glm::length(gizmoWorldPos - camPos);
	float thickness = (0.35f + 0.4f * gizmoObjectRadius) * (1.0f + dist * 0.05f);

	auto raySegmentHit = [&](const glm::vec3& segA, const glm::vec3& segB) -> float {
		glm::vec3 M = segB - segA;
		float lenSq = glm::dot(M, M);
		if (lenSq < 1e-12f) return -1.0f;
		glm::vec3 V = (segA - rayOrig) - glm::dot(segA - rayOrig, rayDir) * rayDir;
		glm::vec3 W = M - glm::dot(M, rayDir) * rayDir;
		float wSq = glm::dot(W, W);
		if (wSq < 1e-12f) {
			float s = 0.5f;
			glm::vec3 P = segA + s * M;
			float tRay = glm::dot(P - rayOrig, rayDir);
			if (tRay <= 0.0f) return -1.0f;
			float d = glm::length(P - (rayOrig + tRay * rayDir));
			return (d <= thickness) ? tRay : -1.0f;
		}
		float s = glm::clamp(-glm::dot(V, W) / wSq, 0.0f, 1.0f);
		glm::vec3 P = segA + s * M;
		float tRay = glm::dot(P - rayOrig, rayDir);
		if (tRay <= 0.0f) return -1.0f;
		float d = glm::length(P - (rayOrig + tRay * rayDir));
		return (d <= thickness) ? tRay : -1.0f;
	};

	glm::vec3 Ax = gizmoWorldPos + glm::vec3(gizmoBaseOffset, 0, 0);
	glm::vec3 Bx = gizmoWorldPos + glm::vec3(gizmoBaseOffset + gizmoArrowLength, 0, 0);
	glm::vec3 Ay = gizmoWorldPos + glm::vec3(0, gizmoBaseOffset, 0);
	glm::vec3 By = gizmoWorldPos + glm::vec3(0, gizmoBaseOffset + gizmoArrowLength, 0);
	glm::vec3 Az = gizmoWorldPos + glm::vec3(0, 0, gizmoBaseOffset);
	glm::vec3 Bz = gizmoWorldPos + glm::vec3(0, 0, gizmoBaseOffset + gizmoArrowLength);

	float tX = raySegmentHit(Ax, Bx);
	float tY = raySegmentHit(Ay, By);
	float tZ = raySegmentHit(Az, Bz);

	int hit = 0;
	float bestT = 1e30f;
	if (tX > 0.0f && tX < bestT) { bestT = tX; hit = 1; }
	if (tY > 0.0f && tY < bestT) { bestT = tY; hit = 2; }
	if (tZ > 0.0f && tZ < bestT) { bestT = tZ; hit = 3; }
	return hit;
}

void VulkanRender::createShadowMapDescriptors()
{
	VkDescriptorSetLayoutBinding uboBinding = {};
	uboBinding.binding = 0;
	uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboBinding.descriptorCount = 1;
	uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &uboBinding;
	VkResult result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &layoutInfo, nullptr, &shadowMapSetLayout);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create shadow map set layout");

	VkDeviceSize uboSize = sizeof(glm::mat4);
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&shadowMapUniformBuffer, &shadowMapUniformBufferMemory);

	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSize.descriptorCount = 1;
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	result = vkCreateDescriptorPool(mainDevice.logicalDevice, &poolInfo, nullptr, &shadowMapDescriptorPool);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create shadow map descriptor pool");

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = shadowMapDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &shadowMapSetLayout;
	result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &allocInfo, &shadowMapDescriptorSet);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate shadow map descriptor set");

	// Initialize shadow UBO with a valid default so the shadow pass always has a usable matrix
	{
		glm::vec3 defaultEye(0.f, 80.f, 80.f);
		glm::vec3 defaultCenter(0.f, 0.f, 0.f);
		glm::vec3 defaultUp(0.f, 1.f, 0.f);
		glm::mat4 defaultView = glm::lookAt(defaultEye, defaultCenter, defaultUp);
		glm::mat4 defaultProj = glm::ortho(-100.f, 100.f, -100.f, 100.f, 0.1f, 400.f);
		defaultProj[1][1] *= -1.f;
		glm::mat4 defaultViewProj = defaultProj * defaultView;
		void* initData = nullptr;
		vkMapMemory(mainDevice.logicalDevice, shadowMapUniformBufferMemory, 0, sizeof(glm::mat4), 0, &initData);
		memcpy(initData, &defaultViewProj, sizeof(glm::mat4));
		vkUnmapMemory(mainDevice.logicalDevice, shadowMapUniformBufferMemory);
	}
	VkDescriptorBufferInfo bufferInfo = {};
	bufferInfo.buffer = shadowMapUniformBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(glm::mat4);
	VkWriteDescriptorSet write = {};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = shadowMapDescriptorSet;
	write.dstBinding = 0;
	write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	write.descriptorCount = 1;
	write.pBufferInfo = &bufferInfo;
	vkUpdateDescriptorSets(mainDevice.logicalDevice, 1, &write, 0, nullptr);

	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	/* CLAMP_TO_BORDER + white: outside shadow frustum samples 1.0 (far = no shadow) */
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	result = vkCreateSampler(mainDevice.logicalDevice, &samplerInfo, nullptr, &shadowMapSampler);
	if (result != VK_SUCCESS) throw std::runtime_error("Failed to create shadow map sampler");
}

void VulkanRender::updateShadowMapUniformBuffer(uint32_t imageIndex)
{
	(void)imageIndex;
	void* data;
	vkMapMemory(mainDevice.logicalDevice, shadowMapUniformBufferMemory, 0, sizeof(glm::mat4), 0, &data);
	memcpy(data, &lightData.lightViewProj, sizeof(glm::mat4));
	// Ensure GPU sees the write (some drivers need explicit flush even with HOST_COHERENT)
	VkMappedMemoryRange range = {};
	range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	range.memory = shadowMapUniformBufferMemory;
	range.offset = 0;
	range.size = sizeof(glm::mat4);
	vkFlushMappedMemoryRanges(mainDevice.logicalDevice, 1, &range);
	vkUnmapMemory(mainDevice.logicalDevice, shadowMapUniformBufferMemory);
}

void VulkanRender::createCommandPool()
{
	QueueFamilyIndices queueFamilyIndices = getQueueFamilyIndices(mainDevice.physicalDevice);
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

	VkResult result = vkCreateCommandPool(mainDevice.logicalDevice, &poolInfo, nullptr, &graphicsCommandPool);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a Command Pool");
	}
}

void VulkanRender::createCommandBuffers()
{
	commandBuffers.resize(swapChainFramebuffers.size());

	VkCommandBufferAllocateInfo cbAllocInfo = {};
	cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbAllocInfo.commandPool = graphicsCommandPool;
	cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbAllocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

	VkResult result = vkAllocateCommandBuffers(mainDevice.logicalDevice, &cbAllocInfo, commandBuffers.data());
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate command buffers");
	}
}

void VulkanRender::createSynchronisation()
{
	// One acquire semaphore (we don't know image index until after acquire). Per-image signal semaphore and fence (fixes validation errors).
	const size_t n = swapChainImages.size();
	imageAvailable.resize(1);
	renderFinished.resize(n);
	drawFences.resize(n);

	VkSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo, nullptr, &imageAvailable[0]) != VK_SUCCESS)
		throw std::runtime_error("Failed to create image available semaphore");

	for (size_t i = 0; i < n; i++) {
		if (vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo, nullptr, &renderFinished[i]) != VK_SUCCESS ||
		    vkCreateFence(mainDevice.logicalDevice, &fenceCreateInfo, nullptr, &drawFences[i]) != VK_SUCCESS)
			throw std::runtime_error("Failed to create sync objects");
	}
}

void VulkanRender::createTextureSampler()
{
	VkSamplerCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	createInfo.magFilter = VK_FILTER_LINEAR;
	createInfo.minFilter = VK_FILTER_LINEAR;
	createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	createInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	createInfo.unnormalizedCoordinates = VK_FALSE;
	createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	createInfo.mipLodBias = 0.0f;
	createInfo.minLod = 0.0f;
	createInfo.maxLod = 0.0f;
	createInfo.anisotropyEnable = VK_TRUE;
	createInfo.maxAnisotropy = 16;


	VkResult result = vkCreateSampler(mainDevice.logicalDevice, &createInfo, nullptr, &textureSampler);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a Texture Sampler");
	}


}

void VulkanRender::createLightUniformBuffers()
{
	VkDeviceSize lBufferSize = sizeof(LightUbo);

	lUniformBuffer.resize(swapChainImages.size());
	lUniformBufferMemory.resize(swapChainImages.size());


	for (size_t i = 0; i < swapChainImages.size(); i++) {
		createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, lBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &lUniformBuffer[i], &lUniformBufferMemory[i]);
	}
}

void VulkanRender::createUniformBuffers()
{
	VkDeviceSize vpBufferSize = sizeof(uboViewProjection);

	//VkDeviceSize modelBufferSize = modelUniformAligment * MAX_OBJECTS;

	vpUniformBuffer.resize(swapChainImages.size());
	vpUniformBufferMemory.resize(swapChainImages.size());

	//modelUniformBuffer.resize(swapChainImages.size());
	//modelUniformBufferMemory.resize(swapChainImages.size());

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, vpBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vpUniformBuffer[i], &vpUniformBufferMemory[i]);
		//createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, modelBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &modelUniformBuffer[i], &modelUniformBufferMemory[i]);
	}
}

void VulkanRender::createDescriptorPool()
{
	VkDescriptorPoolSize uniformPoolSize = {};
	uniformPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uniformPoolSize.descriptorCount = static_cast<uint32_t>(vpUniformBuffer.size()) + static_cast<uint32_t>(lUniformBuffer.size());

	



	VkDescriptorPoolSize shadowSamplerPoolSize = {};
	shadowSamplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	shadowSamplerPoolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());
	std::vector<VkDescriptorPoolSize> poolSizes = { uniformPoolSize, shadowSamplerPoolSize };

	VkDescriptorPoolCreateInfo poolCreateInfo = {};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
	poolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolCreateInfo.pPoolSizes = poolSizes.data();

	VkResult result = vkCreateDescriptorPool(mainDevice.logicalDevice, &poolCreateInfo, nullptr, &descriptorPool);

	if (result != VK_SUCCESS)
		throw std::runtime_error("failed to create a descriptor pool");

	VkDescriptorPoolSize samplerPoolSize = {};
	samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerPoolSize.descriptorCount = MAX_TEXTURES;

	VkDescriptorPoolCreateInfo samplerPoolCreateInfo = {};
	samplerPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	samplerPoolCreateInfo.maxSets = MAX_TEXTURES;
	samplerPoolCreateInfo.poolSizeCount = 1;
	samplerPoolCreateInfo.pPoolSizes = &samplerPoolSize;

	result = vkCreateDescriptorPool(mainDevice.logicalDevice, &samplerPoolCreateInfo, nullptr, &samplerDescriptorPool);
	if (result != VK_SUCCESS)
		throw std::runtime_error("failed to create a descriptor pool");

	VkDescriptorPoolSize colorInputPoolSize = {};
	colorInputPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	colorInputPoolSize.descriptorCount = static_cast<uint32_t>(colourBufferImageView.size());

	VkDescriptorPoolSize depthInputPoolSize = {};
	depthInputPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	depthInputPoolSize.descriptorCount = static_cast<uint32_t>(depthBufferImageView.size());

	std::vector<VkDescriptorPoolSize> inputPoolSizes = { colorInputPoolSize,depthInputPoolSize };

	VkDescriptorPoolCreateInfo inputPoolCreateInfo = {};
	inputPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	inputPoolCreateInfo.maxSets = swapChainImages.size();
	inputPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(inputPoolSizes.size());
	inputPoolCreateInfo.pPoolSizes = inputPoolSizes.data();


	result = vkCreateDescriptorPool(mainDevice.logicalDevice, &inputPoolCreateInfo, nullptr, &inputDescriptorPool);
	if (result != VK_SUCCESS)
		throw std::runtime_error("failed to create a descriptor pool");





}

void VulkanRender::createDescriptorSets()
{
	descriptorSets.resize(swapChainImages.size());

	std::vector<VkDescriptorSetLayout> setayouts(swapChainImages.size(), descriptorSetLayout);

	VkDescriptorSetAllocateInfo setAllocInfo = {};
	setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocInfo.descriptorPool = descriptorPool;
	setAllocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
	setAllocInfo.pSetLayouts = setayouts.data();

	VkResult result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, descriptorSets.data());

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate deescriptor sets");
	}

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		VkDescriptorBufferInfo vpBufferInfo = {};
		vpBufferInfo.buffer = vpUniformBuffer[i];
		vpBufferInfo.offset = 0;
		vpBufferInfo.range = sizeof(uboViewProjection);

		VkWriteDescriptorSet vpSetWrite = {};

		vpSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		vpSetWrite.dstSet = descriptorSets[i];
		vpSetWrite.dstBinding = 0;
		vpSetWrite.dstArrayElement = 0;
		vpSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		vpSetWrite.descriptorCount = 1;
		vpSetWrite.pBufferInfo = &vpBufferInfo;

		VkDescriptorBufferInfo lBufferInfo = {};
		lBufferInfo.buffer = lUniformBuffer[i];
		lBufferInfo.offset = 0;
		lBufferInfo.range = sizeof(LightUbo);

		VkWriteDescriptorSet lSetWrite = {};

		lSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		lSetWrite.dstSet = descriptorSets[i];
		lSetWrite.dstBinding = 1;
		lSetWrite.dstArrayElement = 0;
		lSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		lSetWrite.descriptorCount = 1;
		lSetWrite.pBufferInfo = &lBufferInfo;

		VkDescriptorImageInfo shadowImageInfo = {};
		shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		shadowImageInfo.imageView = shadowMapImageView;
		shadowImageInfo.sampler = shadowMapSampler;

		VkWriteDescriptorSet shadowSetWrite = {};
		shadowSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		shadowSetWrite.dstSet = descriptorSets[i];
		shadowSetWrite.dstBinding = 2;
		shadowSetWrite.dstArrayElement = 0;
		shadowSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		shadowSetWrite.descriptorCount = 1;
		shadowSetWrite.pImageInfo = &shadowImageInfo;

		std::vector<VkWriteDescriptorSet> setWrites = { vpSetWrite, lSetWrite, shadowSetWrite };

		vkUpdateDescriptorSets(mainDevice.logicalDevice, static_cast<uint32_t>(setWrites.size()), setWrites.data(), 0, nullptr);


	}
}

void VulkanRender::createInputDescriptorSets()
{
	inputDescriptorSets.resize(swapChainImages.size());

	std::vector<VkDescriptorSetLayout> setLayouts(swapChainImages.size(), inputSetLayout);

	VkDescriptorSetAllocateInfo setAllocInfo = {};
	setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocInfo.descriptorPool = inputDescriptorPool;
	setAllocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
	setAllocInfo.pSetLayouts = setLayouts.data();

	VkResult result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, inputDescriptorSets.data());
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate input attachment descriptor ssts!");
	}

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		VkDescriptorImageInfo colorAttachmentDescriptor = {};
		colorAttachmentDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		colorAttachmentDescriptor.imageView = colourBufferImageView[i];
		colorAttachmentDescriptor.sampler = VK_NULL_HANDLE;

		VkWriteDescriptorSet colorWrite = {};
		colorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		colorWrite.dstSet = inputDescriptorSets[i];
		colorWrite.dstBinding = 0;
		colorWrite.dstArrayElement = 0;
		colorWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		colorWrite.descriptorCount = 1;
		colorWrite.pImageInfo = &colorAttachmentDescriptor;

		VkDescriptorImageInfo depthAttachmentDescriptor = {};
		depthAttachmentDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		depthAttachmentDescriptor.imageView = depthBufferImageView[i];
		depthAttachmentDescriptor.sampler = VK_NULL_HANDLE;

		VkWriteDescriptorSet depthWrite = {};
		depthWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		depthWrite.dstSet = inputDescriptorSets[i];
		depthWrite.dstBinding = 1;
		depthWrite.dstArrayElement = 0;
		depthWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		depthWrite.descriptorCount = 1;
		depthWrite.pImageInfo = &depthAttachmentDescriptor;


		std::vector<VkWriteDescriptorSet> setWrites = { colorWrite,depthWrite };

		vkUpdateDescriptorSets(mainDevice.logicalDevice, static_cast<uint32_t>(setWrites.size()), setWrites.data(), 0, nullptr);
	}
}

void VulkanRender::updateUniformBuffers(uint32_t imageIndex)
{
	uboViewProjection.view = cam;
	void* data;
	vkMapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex], 0, sizeof(UboViewProjection), 0, &data);
	memcpy(data, &uboViewProjection, sizeof(UboViewProjection));
	vkUnmapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex]);

	// Directional/yildiz: orthographic shadow camera (yildiz konumundan sahneye bakis)
	const glm::vec3 sceneCenter = shadowSceneCenter;
	const float shadowNear = 0.01f;
	const float shadowFar = 300.0f;
	const float orthoHalf = 25.0f;
	glm::vec3 lightEye;
	glm::vec3 sunDir;
	if (lightData.starPosition.w != 0.0f) {
		// Yildiz isik kaynagi: kamera yildizda, sahne merkezine bak
		lightEye = glm::vec3(lightData.starPosition);
		sunDir = glm::normalize(sceneCenter - lightEye);
	} else {
		sunDir = glm::normalize(glm::vec3(lightData.direction));
		const float shadowDistance = 80.0f;
		lightEye = sceneCenter + sunDir * shadowDistance;
	}
	glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
	glm::vec3 up = worldUp;
	if (std::abs(glm::dot(sunDir, worldUp)) > 0.99f)
		up = glm::vec3(0.0f, 0.0f, 1.0f);
	glm::mat4 lightView = glm::lookAt(lightEye, sceneCenter, up);
	glm::mat4 lightProj = glm::ortho(-orthoHalf, orthoHalf, -orthoHalf, orthoHalf, shadowNear, shadowFar);
	lightData.lightViewProj = lightProj * lightView;
	updateShadowMapUniformBuffer(imageIndex);

	/*
	for (size_t i = 0; i < meshList.size(); i++) {
		UboModel* thisModel = ( UboModel* )((uint64_t)modelTransferSpace + (i * modelUniformAligment));
		*thisModel = meshList[i].getModel();
	}

	vkMapMemory(mainDevice.logicalDevice, modelUniformBufferMemory[imageIndex], 0, modelUniformAligment * meshList.size(), 0, &data);
	memcpy(data, modelTransferSpace, modelUniformAligment * meshList.size());
	vkUnmapMemory(mainDevice.logicalDevice, modelUniformBufferMemory[imageIndex]);
	*/


	// Bellek Kopyalama
	vkMapMemory(mainDevice.logicalDevice, lUniformBufferMemory[imageIndex], 0, sizeof(LightUbo), 0, &data);
	memcpy(data, &lightData, sizeof(LightUbo));
	vkUnmapMemory(mainDevice.logicalDevice, lUniformBufferMemory[imageIndex]);
}

void VulkanRender::recordCommands(uint32_t currentImage)
{
	static bool firstShadowPass = true;
	VkCommandBuffer cmd = commandBuffers[currentImage];

	VkCommandBufferBeginInfo bufferBeginInfo = {};
	bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	VkResult result = vkBeginCommandBuffer(cmd, &bufferBeginInfo);
	if (result != VK_SUCCESS)
		throw std::runtime_error("Failed to start recording a command buffer");

	// Ensure shadow UBO (written on CPU in updateUniformBuffers) is visible to the shadow pass
	VkMemoryBarrier uboBarrier = {};
	uboBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	uboBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
	uboBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &uboBarrier, 0, nullptr, 0, nullptr);

	VkImageAspectFlags shadowAspect = (shadowMapDepthFormat == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT : (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
	// Transition shadow map for shadow pass (first time: UNDEFINED -> ATTACHMENT; later: READ_ONLY -> ATTACHMENT)
	VkImageMemoryBarrier shadowBarrier = {};
	shadowBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	shadowBarrier.oldLayout = firstShadowPass ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
	shadowBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	shadowBarrier.image = shadowMapImage;
	shadowBarrier.subresourceRange.aspectMask = shadowAspect;
	shadowBarrier.subresourceRange.baseMipLevel = 0;
	shadowBarrier.subresourceRange.levelCount = 1;
	shadowBarrier.subresourceRange.baseArrayLayer = 0;
	shadowBarrier.subresourceRange.layerCount = 1;
	shadowBarrier.srcAccessMask = firstShadowPass ? 0 : VK_ACCESS_SHADER_READ_BIT;
	shadowBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	VkPipelineStageFlags srcStage = firstShadowPass ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &shadowBarrier);
	firstShadowPass = false;

	// ---- Shadow pass: render scene from light's view into shadow map depth ----
	VkClearValue shadowClear = {};
	shadowClear.depthStencil.depth = 1.0f;
	VkRenderPassBeginInfo shadowPassBegin = {};
	shadowPassBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	shadowPassBegin.renderPass = shadowMapRenderPass;
	shadowPassBegin.framebuffer = shadowMapFramebuffer;
	shadowPassBegin.renderArea = { { 0, 0 }, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE } };
	shadowPassBegin.clearValueCount = 1;
	shadowPassBegin.pClearValues = &shadowClear;
	vkCmdBeginRenderPass(cmd, &shadowPassBegin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipelineLayout, 0, 1, &shadowMapDescriptorSet, 0, nullptr);
	glm::vec3 shadowLightPos(0.0f);
	bool shadowUseStar = (lightData.starPosition.w != 0.0f);
	if (shadowUseStar)
		shadowLightPos = glm::vec3(lightData.starPosition);

	for (size_t j = 0; j < modelList.size(); j++) {
		if (j < modelVisible.size() && !modelVisible[j])
			continue;  /* sahne objesi kaldirildi, cizme */
		if (lightSpriteModelId >= 0 && j == static_cast<size_t>(lightSpriteModelId))
			continue;  /* don't cast shadow from the light sprite quad */
		if (floorModelId >= 0 && j == static_cast<size_t>(floorModelId))
			continue;  /* floor only receives shadows, does not cast (avoids huge rect shadow) */
		if (gizmoArrowModelId >= 0 && j == static_cast<size_t>(gizmoArrowModelId))
			continue;  /* gizmo arrow is UI, no shadow */
		MeshModel& thisModel = modelList[j];
		glm::mat4 modelMatrix = thisModel.getModel();
		/* Isik kaynaginin arkasindaki objeler golge dussun (F1 gunesin ote tarafindayken dunyaya golge vermesin) */
		if (shadowUseStar) {
			glm::vec3 objPos(modelMatrix[3]);
			if (glm::dot(objPos - shadowLightPos, shadowSceneCenter - shadowLightPos) < 0.0f)
				continue;  /* obje isigin arkasinda, golge haritasina cizme */
		}
		for (size_t k = 0; k < thisModel.getMeshCount(); k++) {
			vkCmdPushConstants(cmd, shadowMapPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &modelMatrix);
			VkBuffer vertexBuffers[] = { thisModel.getMesh(k)->getVertexBuffer() };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(cmd, thisModel.getMesh(k)->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, thisModel.getMesh(k)->getIndexCount(), 1, 0, 0, 0);
		}
	}
	vkCmdEndRenderPass(cmd);
	// Render pass finalLayout already transitioned shadow map to READ_ONLY for main pass.

	// ---- Main pass ----
	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset = { 0,0 };
	renderPassBeginInfo.renderArea.extent = swapChainExtent;
	std::array<VkClearValue, 2> clearValues = {};
	clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[1].depthStencil.depth = 1.0f;
	renderPassBeginInfo.pClearValues = clearValues.data();
	renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.framebuffer = swapChainFramebuffers[currentImage];

	vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	// Sky (gradient): draw first so scene is on top; disabled by default (SetSkyEnabled(true) to re-enable)
	if (skyEnabled_ && skyPipeline != VK_NULL_HANDLE && skyPipelineLayout != VK_NULL_HANDLE) {
		glm::mat4 vp = uboViewProjection.projection * uboViewProjection.view;
		glm::mat4 invViewProj = glm::inverse(vp);
		glm::vec3 cameraPos = glm::vec3(glm::inverse(uboViewProjection.view)[3]);
		struct SkyPush {
			glm::mat4 invViewProj;
			glm::vec3 cameraPos; float pad0;
			glm::vec3 skyTop;    float pad1;
			glm::vec3 skyBottom; float pad2;
		} skyPush;
		skyPush.invViewProj = invViewProj;
		skyPush.cameraPos = cameraPos;
		skyPush.pad0 = 0.0f;
		skyPush.skyTop = glm::vec3(0.35f, 0.55f, 0.95f);
		skyPush.pad1 = 0.0f;
		skyPush.skyBottom = glm::vec3(0.65f, 0.78f, 0.92f);
		skyPush.pad2 = 0.0f;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
		vkCmdPushConstants(cmd, skyPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(skyPush), &skyPush);
		vkCmdDraw(cmd, 3, 1, 0, 0);
	}

	if (starPipeline != VK_NULL_HANDLE && starPipelineLayout != VK_NULL_HANDLE && starVertexBuffer != VK_NULL_HANDLE && starIndexBuffer != VK_NULL_HANDLE) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, starPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, starPipelineLayout, 0, 1, &descriptorSets[currentImage], 0, nullptr);
		vkCmdPushConstants(cmd, starPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(StarParams), &starParams);
		VkBuffer vb[] = { starVertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(cmd, 0, 1, vb, offsets);
		vkCmdBindIndexBuffer(cmd, starIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(cmd, starIndexCount, 1, 0, 0, 0);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
	for (size_t j = 0; j < modelList.size(); j++) {
		if (j < modelVisible.size() && !modelVisible[j])
			continue;  /* sahne objesi kaldirildi, cizme */
		if (gizmoArrowModelId >= 0 && j == static_cast<size_t>(gizmoArrowModelId))
			continue;  /* gizmo arrow drawn separately as 3 instances */
		MeshModel thisModel = modelList[j];
		glm::mat4 modelMatrix = thisModel.getModel();
		vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model), &modelMatrix);

		for (size_t k = 0; k < thisModel.getMeshCount(); k++) {
			VkBuffer vertexBuffers[] = { thisModel.getMesh(k)->getVertexBuffer() };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(cmd, thisModel.getMesh(k)->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

			std::array<VkDescriptorSet, 2> descriptorSetGroup = { descriptorSets[currentImage], samplerDescriptorSets[thisModel.getMesh(k)->getTexId()] };
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, static_cast<uint32_t>(descriptorSetGroup.size()), descriptorSetGroup.data(), 0, nullptr);
			vkCmdDrawIndexed(cmd, thisModel.getMesh(k)->getIndexCount(), 1, 0, 0, 0);
		}
	}

	// XYZ gizmo (secili obje): ok modeli (UIObjects) veya cizgi
	if (lightGizmoSelected) {
		if (gizmoArrowModelId >= 0 && gizmoAxisTexId[0] >= 0 && gizmoAxisTexId[1] >= 0 && gizmoAxisTexId[2] >= 0) {
			MeshModel& arrowModel = modelList[gizmoArrowModelId];
			const float arrowScale = gizmoArrowLength / 3.0f;  // model ~3 birim, objeye gore kucuk
			const glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(arrowScale));
			const glm::mat4 Ry180 = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0));  // ok yonu duz
			const glm::mat4 Rz90 = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0, 0, 1));
			const glm::mat4 Ry90 = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0));
			const glm::vec3 axisX(1,0,0), axisY(0,1,0), axisZ(0,0,1);
			glm::mat4 T[3] = {
				glm::translate(glm::mat4(1.0f), gizmoWorldPos + axisX * gizmoBaseOffset),
				glm::translate(glm::mat4(1.0f), gizmoWorldPos + axisY * gizmoBaseOffset),
				glm::translate(glm::mat4(1.0f), gizmoWorldPos + axisZ * gizmoBaseOffset)
			};
			// X: S only; Y,Z: 90+180. Kirmizi tersse X icin Ry180 ekle veya cikar.
			glm::mat4 modelMatrix[3] = { T[0] * S, T[1] * Rz90 * Ry180 * S, T[2] * Ry90 * Ry180 * S };
			const float highlightScale = 1.45f;  // secili ok biraz daha buyuk
			for (int axis = 0; axis < 3; axis++) {
				bool highlighted = (gizmoHighlightAxis == axis + 1);
				int texId = (highlighted && gizmoHighlightTexId >= 0) ? gizmoHighlightTexId : gizmoAxisTexId[axis];
				if (texId < 0) continue;
				glm::mat4 M = modelMatrix[axis];
				if (highlighted) M = glm::scale(M, glm::vec3(highlightScale));
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model), &M);
				for (size_t k = 0; k < arrowModel.getMeshCount(); k++) {
					VkBuffer vertexBuffers[] = { arrowModel.getMesh(k)->getVertexBuffer() };
					VkDeviceSize offsets[] = { 0 };
					vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
					vkCmdBindIndexBuffer(cmd, arrowModel.getMesh(k)->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);
					std::array<VkDescriptorSet, 2> descriptorSetGroup = { descriptorSets[currentImage], samplerDescriptorSets[texId] };
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, static_cast<uint32_t>(descriptorSetGroup.size()), descriptorSetGroup.data(), 0, nullptr);
					vkCmdDrawIndexed(cmd, arrowModel.getMesh(k)->getIndexCount(), 1, 0, 0, 0);
				}
			}
		} else if (gizmoLinePipeline != VK_NULL_HANDLE) {
			glm::mat4 vp = uboViewProjection.projection * uboViewProjection.view;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gizmoLinePipeline);
			vkCmdPushConstants(cmd, gizmoLinePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &vp);
			VkDeviceSize off = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &gizmoVertexBuffer, &off);
			vkCmdBindIndexBuffer(cmd, gizmoIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, 18, 1, 0, 0, 0);
		}
	}

	vkCmdEndRenderPass(cmd);
	result = vkEndCommandBuffer(cmd);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to stop recording a command");
	}

}

void VulkanRender::allocateDynamicBufferTransferSpace()
{
	/*

	modelUniformAligment = sizeof(PushModel) + minUniformBufferOffset-1 & ~(minUniformBufferOffset - 1);

	modelTransferSpace = (PushModel*)_aligned_malloc(modelUniformAligment * MAX_OBJECTS, modelUniformAligment);
	*/
}

void VulkanRender::getPhysicalDevice()
{
	uint32_t physicalDeviceCount = 0;

	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

	if (physicalDeviceCount == 0) {
		std::runtime_error("Cant find device");
	}

	std::vector<VkPhysicalDevice> deviceList(physicalDeviceCount);

	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, deviceList.data());

	for (const auto& device : deviceList)
	{
		if (checkDeviceSuitable(device))
		{
			mainDevice.physicalDevice = device;
			break;
		}
	}

	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(mainDevice.physicalDevice, &deviceProperties);

	//minUniformBufferOffset = deviceProperties.limits.minUniformBufferOffsetAlignment;
}

std::vector<const char*> VulkanRender::getRequiredExtensions()
{
	std::vector<const char*> instanceExtensions = std::vector<const char*>();

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;

	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);


	for (size_t i = 0; i < glfwExtensionCount; i++) {
		instanceExtensions.push_back(glfwExtensions[i]);
	}

	if (!checkInstanceExtensionSupport(&instanceExtensions)) {
		throw std::runtime_error("vk instance does not support required extensions!");
	}

	if (enableValidationLayers) {
		instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return instanceExtensions;
}

bool VulkanRender::checkInstanceExtensionSupport(std::vector<const char*>* checkExtensions)
{
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());


	for (const auto& checkExtensions : *checkExtensions) {
		bool hasExtension = false;
		for (const auto& extensions : extensions) {
			if (strcmp(checkExtensions, extensions.extensionName)) {
				hasExtension = true;
				break;
			}
		}

		if (!hasExtension) {
			return false;
		}
	}


	return true;
}

bool VulkanRender::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
	uint32_t extensionCount = 0;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	if (extensionCount == 0) {
		return false;
	}

	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

	for (const auto& deviceExtensions : deviceExtensions) {
		bool hasExtension = false;
		for (const auto& extension : extensions) {
			if (strcmp(deviceExtensions, extension.extensionName) == 0) {
				hasExtension = true;
				break;
			}
		}

		if (!hasExtension) {
			return false;
		}
	}
}

bool VulkanRender::checkDeviceSuitable(VkPhysicalDevice device)
{
	/*
	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(device,&physicalDeviceProperties);
	*/

	VkPhysicalDeviceFeatures physicalDeviceFeatures;
	vkGetPhysicalDeviceFeatures(device, &physicalDeviceFeatures);


	QueueFamilyIndices indices = getQueueFamilyIndices(device);

	bool extensionsSupported = checkDeviceExtensionSupport(device);

	bool swapChainValid = false;
	if (extensionsSupported) {
		SwapChainDetails swapChainDetails = getSwapChainDetails(device);
		swapChainValid = !swapChainDetails.presentModes.empty() && !swapChainDetails.formats.empty();
	}


	return indices.isValid() && extensionsSupported && swapChainValid && physicalDeviceFeatures.samplerAnisotropy;
}

bool VulkanRender::checkValidationLayerSupport()
{
	uint32_t layerCount = 0;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());



	for (const auto* layerName : validationLayers) {
		bool layerFound = false;
		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
			return false;
	}
	return true;
}

QueueFamilyIndices VulkanRender::getQueueFamilyIndices(VkPhysicalDevice device)
{
	QueueFamilyIndices indices;

	uint32_t queueCount = 0;

	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queueFamilyProperties.data());


	int i = 0;

	for (const auto& queueFamily : queueFamilyProperties) {
		if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.graphicsFamily = i;
		}

		VkBool32 presentationSupport = false;

		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentationSupport);

		if (queueFamily.queueCount > 0 && presentationSupport) {
			indices.presentationFamily = i;
		}

		if (indices.isValid()) {
			break;
		}

		i++;
	}



	return indices;
}

SwapChainDetails VulkanRender::getSwapChainDetails(VkPhysicalDevice device)
{
	SwapChainDetails swapChainDetails;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &swapChainDetails.surfaceCapabilities);

	uint32_t formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		swapChainDetails.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, swapChainDetails.formats.data());
	}


	uint32_t presentCount = 0;

	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentCount, nullptr);

	if (presentCount != 0) {
		swapChainDetails.presentModes.resize(presentCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentCount, swapChainDetails.presentModes.data());
	}



	return swapChainDetails;
}


// format : VK_FORMAT_R8G8B8A8_UNORM
// colorSpace : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
VkSurfaceFormatKHR VulkanRender::chooseBestSurfaceFromat(const std::vector<VkSurfaceFormatKHR>& formats)
{
	if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
		return { VK_FORMAT_R8G8B8A8_UNORM ,VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}

	for (const auto& format : formats) {
		if ((format.format == VK_FORMAT_R8G8B8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_UNORM) && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return format;
		}
	}

	return formats[0];
}

VkPresentModeKHR VulkanRender::chooseBestPresentationMode(const std::vector<VkPresentModeKHR> presentationModes)
{
	for (const auto& presentationMode : presentationModes) {
		if (presentationMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return presentationMode;
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanRender::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities)
{
	if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return surfaceCapabilities.currentExtent;
	}
	else {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D newExtent = {};
		newExtent.width = static_cast<uint32_t>(width);
		newExtent.height = static_cast<uint32_t>(height);


		newExtent.width = std::max(surfaceCapabilities.minImageExtent.width, std::min(surfaceCapabilities.maxImageExtent.width, newExtent.width));
		newExtent.height = std::max(surfaceCapabilities.minImageExtent.height, std::min(surfaceCapabilities.maxImageExtent.height, newExtent.height));

		return newExtent;
	}
}

VkFormat VulkanRender::chooseSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags featureFlags)
{
	for (VkFormat format : formats) {
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(mainDevice.physicalDevice, format, &properties);

		if (tiling == VK_IMAGE_TILING_LINEAR && (properties.linearTilingFeatures & featureFlags) == featureFlags) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & featureFlags) == featureFlags) {
			return format;
		}
	}

	throw std::runtime_error("Failed to find matching format");
}

VkImage VulkanRender::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags useFlags, VkMemoryPropertyFlags propFlags, VkDeviceMemory* imageMemory)
{
	VkImageCreateInfo imageCreateInfo = {};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.extent.width = width;
	imageCreateInfo.extent.height = height;
	imageCreateInfo.extent.depth = 1;
	imageCreateInfo.mipLevels = 1;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.format = format;
	imageCreateInfo.tiling = tiling;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.usage = useFlags;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkImage image;
	VkResult result = vkCreateImage(mainDevice.logicalDevice, &imageCreateInfo, nullptr, &image);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a vk image");
	}

	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(mainDevice.logicalDevice, image, &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocInfo = {};
	memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocInfo.allocationSize = memoryRequirements.size;
	memoryAllocInfo.memoryTypeIndex = findMemoryTypeIndex(mainDevice.physicalDevice, memoryRequirements.memoryTypeBits, propFlags);

	result = vkAllocateMemory(mainDevice.logicalDevice, &memoryAllocInfo, nullptr, imageMemory);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate memory for iamge!");
	}
	vkBindImageMemory(mainDevice.logicalDevice, image, *imageMemory, 0);
	return image;
}

VkImageView VulkanRender::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
	VkImageViewCreateInfo viewCreateInfo = {};
	viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCreateInfo.image = image;
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewCreateInfo.format = format;
	viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;


	viewCreateInfo.subresourceRange.aspectMask = aspectFlags;
	viewCreateInfo.subresourceRange.baseMipLevel = 0;
	viewCreateInfo.subresourceRange.levelCount = 1;
	viewCreateInfo.subresourceRange.baseArrayLayer = 0;
	viewCreateInfo.subresourceRange.layerCount = 1;


	VkImageView imageView;

	VkResult result = vkCreateImageView(mainDevice.logicalDevice, &viewCreateInfo, nullptr, &imageView);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create an Image View!");
	}

	return imageView;
}

VkShaderModule VulkanRender::createShaderModule(const std::vector<char>& code)
{
	VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
	shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModuleCreateInfo.codeSize = code.size();
	shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	VkResult result = vkCreateShaderModule(mainDevice.logicalDevice, &shaderModuleCreateInfo, nullptr, &shaderModule);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to create a shader module");
	}
	return shaderModule;
}

/* SCENE FOLDER SUPPORT: createTextureImage passes textureBasePath to loadTextureFile. */
int VulkanRender::createTextureImage(std::string fileName, std::string textureBasePath)
{
	int width, height;
	VkDeviceSize imageSize;
	stbi_uc* imageData = loadTextureFile(fileName, &width, &height, &imageSize, textureBasePath);

	VkBuffer imageStagingBuffer;
	VkDeviceMemory imageStagingBufferMemory;
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &imageStagingBuffer, &imageStagingBufferMemory);

	void* data;
	vkMapMemory(mainDevice.logicalDevice, imageStagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, imageData, static_cast<size_t>(imageSize));
	vkUnmapMemory(mainDevice.logicalDevice, imageStagingBufferMemory);

	stbi_image_free(imageData);

	VkImage texImage;
	VkDeviceMemory texImageMemory;

	texImage = createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &texImageMemory);


	transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	copyImageBuffer(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, imageStagingBuffer, texImage, width, height);

	transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	textureImages.push_back(texImage);
	textureImageMemory.push_back(texImageMemory);

	vkDestroyBuffer(mainDevice.logicalDevice, imageStagingBuffer, nullptr);
	vkFreeMemory(mainDevice.logicalDevice, imageStagingBufferMemory, nullptr);
	return textureImages.size() - 1;
}

/** Creates a 1x1 solid-color texture. Returns texture descriptor id for use with meshes. */
int VulkanRender::createSolidColorTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	const uint32_t width = 1, height = 1;
	const VkDeviceSize imageSize = width * height * 4;
	uint8_t pixel[4] = { r, g, b, a };

	VkBuffer imageStagingBuffer;
	VkDeviceMemory imageStagingBufferMemory;
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &imageStagingBuffer, &imageStagingBufferMemory);
	void* data;
	vkMapMemory(mainDevice.logicalDevice, imageStagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, pixel, static_cast<size_t>(imageSize));
	vkUnmapMemory(mainDevice.logicalDevice, imageStagingBufferMemory);

	VkImage texImage;
	VkDeviceMemory texImageMemory;
	texImage = createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &texImageMemory);
	transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copyImageBuffer(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, imageStagingBuffer, texImage, width, height);
	transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	textureImages.push_back(texImage);
	textureImageMemory.push_back(texImageMemory);
	vkDestroyBuffer(mainDevice.logicalDevice, imageStagingBuffer, nullptr);
	vkFreeMemory(mainDevice.logicalDevice, imageStagingBufferMemory, nullptr);

	VkImageView imageView = createImageView(textureImages.back(), VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
	textureImageViews.push_back(imageView);
	return createTextureDescriptor(imageView);
}

/* SCENE FOLDER SUPPORT: createTexture accepts textureBasePath for models in Scene/ or other folders. */
int VulkanRender::createTexture(std::string fileName, std::string textureBasePath)
{
	int textureImageLoc = createTextureImage(fileName, textureBasePath);

	VkImageView imageView = createImageView(textureImages[textureImageLoc], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
	textureImageViews.push_back(imageView);

	int descriptorLoc = createTextureDescriptor(imageView);

	return descriptorLoc;
}

int VulkanRender::createTextureDescriptor(VkImageView textureImage)
{
	VkDescriptorSet descriptorSet = {};

	VkDescriptorSetAllocateInfo setAllocInfo = {};
	setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocInfo.descriptorPool = samplerDescriptorPool;
	setAllocInfo.descriptorSetCount = 1;
	setAllocInfo.pSetLayouts = &samplerSetLayout;


	VkResult result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, &descriptorSet);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor set");
	}

	VkDescriptorImageInfo imageInfo = {};
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	imageInfo.imageView = textureImage;
	imageInfo.sampler = textureSampler;


	VkWriteDescriptorSet desciiptorWrite = {};
	desciiptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	desciiptorWrite.dstSet = descriptorSet;
	desciiptorWrite.dstBinding = 0;
	desciiptorWrite.dstArrayElement = 0;
	desciiptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	desciiptorWrite.descriptorCount = 1;
	desciiptorWrite.pImageInfo = &imageInfo;

	vkUpdateDescriptorSets(mainDevice.logicalDevice, 1, &desciiptorWrite, 0, nullptr);

	samplerDescriptorSets.push_back(descriptorSet);
	return samplerDescriptorSets.size() - 1;

}

/*
 * SCENE FOLDER SUPPORT: createMeshModel derives texture base path from model path.
 * Example: "Scene/Sponza.gltf" -> textures loaded from "Scene/" first, then "Textures/" fallback.
 */
int VulkanRender::createMeshModel(std::string modelFile)
{
	/* Derive texture base path from model file directory so Scene/Sponza.gltf uses Scene/ for textures. */
	size_t lastSlash = modelFile.find_last_of("/\\");
	std::string textureBasePath = (lastSlash != std::string::npos) ? modelFile.substr(0, lastSlash + 1) : "Textures/";

	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(modelFile, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);
	if (!scene) {
		std::string err = std::string("Failed to load Model: ") + modelFile + " (dosya yok veya Assimp hatasi: " + importer.GetErrorString() + ")";
		throw std::runtime_error(err);
	}

	std::vector<std::string> textureNames = MeshModel::loadMaterials(scene);
	std::vector<int> matToTex(textureNames.size());

	for (size_t i = 0; i < textureNames.size(); i++) {
		if (textureNames[i].empty()) {
			matToTex[i] = 0;
		}
		else {
			try {
				/* Pass texture base path so Scene folder textures are found. */
				matToTex[i] = createTexture(textureNames[i], textureBasePath);
			}
			catch (const std::exception& e) {
				std::cerr << "Texture load failed for '" << textureNames[i] << "', using default. " << e.what() << std::endl;
				matToTex[i] = 0;
			}
		}
	}

	std::vector<Mesh> modelMeshes = MeshModel::LoadNode(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, scene->mRootNode, scene, matToTex);

	MeshModel meshModel = MeshModel(modelMeshes);
	modelList.push_back(meshModel);
	modelVisible.push_back(true);

	return modelList.size() - 1;
}

int VulkanRender::createQuadMeshModel(int textureId)
{
	const float s = 1.0f;  // quad half-size (world units)
	std::vector<Vertex> verts = {
		{ {1,1,1}, { -s, -s, 0 }, { 0, 0 }, { 0, 0, 1 } },
		{ {1,1,1}, {  s, -s, 0 }, { 1, 0 }, { 0, 0, 1 } },
		{ {1,1,1}, {  s,  s, 0 }, { 1, 1 }, { 0, 0, 1 } },
		{ {1,1,1}, { -s,  s, 0 }, { 0, 1 }, { 0, 0, 1 } }
	};
	std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };
	Mesh quadMesh(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, &verts, &indices, textureId);
	MeshModel quadModel(std::vector<Mesh>{ quadMesh });
	modelList.push_back(quadModel);
	modelVisible.push_back(true);
	return static_cast<int>(modelList.size() - 1);
}

int VulkanRender::createFloorMeshModel(float halfSize, int textureId)
{
	// Large quad in XZ plane (y = 0), normal +Y for receiving shadows; white vertex color
	const glm::vec3 color(1.0f, 1.0f, 1.0f);
	const glm::vec3 n(0.0f, 1.0f, 0.0f);
	std::vector<Vertex> verts = {
		{ color, { -halfSize, 0.0f, -halfSize }, { 0, 0 }, n },
		{ color, {  halfSize, 0.0f, -halfSize }, { 1, 0 }, n },
		{ color, {  halfSize, 0.0f,  halfSize }, { 1, 1 }, n },
		{ color, { -halfSize, 0.0f,  halfSize }, { 0, 1 }, n }
	};
	// Winding so top face (+Y) is front-face when viewed from above (counter-clockwise in XZ)
	std::vector<uint32_t> indices = { 0, 3, 2, 0, 2, 1 };
	Mesh floorMesh(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, &verts, &indices, textureId);
	MeshModel floorModel(std::vector<Mesh>{ floorMesh });
	modelList.push_back(floorModel);
	modelVisible.push_back(true);
	return static_cast<int>(modelList.size() - 1);
}

int VulkanRender::createLightSpriteModel()
{
	lightSpriteModelId = createQuadMeshModel(lightSpriteTextureId);
	return lightSpriteModelId;
}

/*
 * SCENE FOLDER SUPPORT: loadTextureFile accepts textureBasePath (e.g. "Scene/" or "Textures/").
 * Tries textureBasePath + fileName first; if that fails and base path is not "Textures/", retries with "Textures/" + fileName.
 */
stbi_uc* VulkanRender::loadTextureFile(std::string fileName, int* width, int* height, VkDeviceSize* imageSize, std::string textureBasePath)
{
	int channels;
	/* Ensure path has trailing slash for concatenation. */
	if (!textureBasePath.empty() && textureBasePath.back() != '/' && textureBasePath.back() != '\\')
		textureBasePath += "/";

	std::string fileLoc = textureBasePath + fileName;
	stbi_uc* image = stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);

	/* Fallback 1: path may be "textures/foo.jpeg" but file is in model folder as "foo.jpeg" */
	if (!image && textureBasePath != "Textures/") {
		size_t lastSlash = fileName.rfind('/');
		if (lastSlash != std::string::npos) {
			std::string baseName = fileName.substr(lastSlash + 1);
			fileLoc = textureBasePath + baseName;
			image = stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);
		}
	}
	/* Fallback 2: try global Textures/ folder */
	if (!image && textureBasePath != "Textures/") {
		fileLoc = "Textures/" + fileName;
		image = stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);
	}
	if (!image && textureBasePath != "Textures/") {
		size_t lastSlash = fileName.rfind('/');
		if (lastSlash != std::string::npos) {
			fileLoc = "Textures/" + fileName.substr(lastSlash + 1);
			image = stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);
		}
	}

	std::cout << "Attempting to load texture from: " << fileLoc << std::endl;
	if (!image) {
		throw std::runtime_error("Failed to load texture file! (" + fileName + ")");
	}

	*imageSize = *width * *height * 4;
	return image;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRender::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

VkResult VulkanRender::createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void VulkanRender::destroyDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

void VulkanRender::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
}