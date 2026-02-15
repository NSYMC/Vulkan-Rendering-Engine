#define STB_IMAGE_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#include <stdexcept>
#include <vector>
#include <iostream>

#include "VulkanRender.h"
#include "Input.h"
#include "Camera.h"

VulkanRender vulkanRenderer;
Input input;
Camera camera;
GLFWwindow* window;
float deltaTime;
float angle;



/// <commands>
bool rotateLeft;
bool rotateRight;

bool lightRight;
bool lightLeft;
bool lightForwad;
bool lightBackward;
bool lightUp;
bool lightDown;
/// <commands>


void setWindow(std::string wName = "Test window", const int width = 1920, const int height = 1080) {
	glfwInit();

	// says glfw to not work with opengl
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(width, height, wName.c_str(),nullptr,nullptr);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int modes) {
	if (key == GLFW_KEY_E && action == GLFW_PRESS) {
		rotateLeft = true;
	}
	if (key == GLFW_KEY_E && action == GLFW_RELEASE) {
		rotateLeft = false;
	}
	if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
		rotateRight = true;
	}
	if (key == GLFW_KEY_Q && action == GLFW_RELEASE) {
		rotateRight = false;
	}
	if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
		lightForwad = true;
	}
	if (key == GLFW_KEY_UP && action == GLFW_RELEASE) {
		lightForwad = false;
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		lightBackward = true;
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_RELEASE) {
		lightBackward = false;
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		lightLeft = true;
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_RELEASE) {
		lightLeft = false;
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		lightRight = true;
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_RELEASE) {
		lightRight = false;
	}
	if (key == GLFW_KEY_O && action == GLFW_PRESS) {
		lightUp = true;
	}
	if (key == GLFW_KEY_O && action == GLFW_RELEASE) {
		lightUp = false;
	}
	if (key == GLFW_KEY_P && action == GLFW_PRESS) {
		lightDown = true;
	}
	if (key == GLFW_KEY_P && action == GLFW_RELEASE) {
		lightDown = false;
	}
}



int main() {

	setWindow();
	camera.input = input;

	

	
	if (vulkanRenderer.init(window) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}

	angle = 0.0f;
	deltaTime = 0.0f;
	float lastTime = 0.0f;

	int f1 = vulkanRenderer.createMeshModel("Models/mcl35m_2.obj");

	//glfwSetKeyCallback(window, key_callback);
	glfwSetKeyCallback(window, Input::KeyCallback);
	float x, y, z;
	x = 0;
	z = 0; 
	y = -5.f;
	LightUbo lightData = {
		{x,y,z,0.0f},
		{1.0f,1.f,1.f,1.f },
		{0.0f,0.f,0.f,0.f}
	};
	

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float now = glfwGetTime();
		deltaTime = now - lastTime;
		lastTime = now;
		if (rotateLeft) {
			angle += 10.f * deltaTime;
		}
		if (rotateRight) {
			angle -= 10.f * deltaTime;
		}

		if (lightBackward) {
			z -= 10.f * deltaTime;
		}
		if (lightForwad) {
			z += 10.f * deltaTime;
		}
		if (lightDown) {
			y += 10.f * deltaTime;
		}
		if (lightUp) {
			y -= 10.f * deltaTime;
		}
		if (lightLeft) {
			x += 10.f * deltaTime;
		}
		if (lightRight) {
			x -= 10.f * deltaTime;
		}

		camera.Update(deltaTime);
		vulkanRenderer.cam = camera.GetViewMatrix();
		glm::mat4 testMat = glm::rotate(glm::mat4(1.0f), glm::radians(angle), glm::vec3(0.0f, 1.0f, 0.0f));
		testMat = glm::rotate(testMat, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		vulkanRenderer.updateModel(f1, testMat);

		lightData.position.x = x;
		lightData.position.y = y;
		lightData.position.z = z;

		vulkanRenderer.updateLight(lightData);

		
		
		vulkanRenderer.draw();

		Input::Update();
	}

	vulkanRenderer.cleanup();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}




