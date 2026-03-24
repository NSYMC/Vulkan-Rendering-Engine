#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include "Input.h"

struct GLFWwindow;

class Camera
{
public:
	Camera();
	glm::mat4 GetViewMatrix();
	void Update(float deltaTime);
	void SetWindow(GLFWwindow* win) { window = win; }
	~Camera();

	glm::vec3 position = glm::vec3(5.0f, 2.0f, 5.0f);
	glm::vec3 front = glm::vec3(0, 0, -1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	float yaw = -90.0f;   // initial look along -Z
	float pitch = 0.0f;
	float movementSpeed = 40.0f;
	float sprintMultiplier = 2.5f;  // speed multiplier when Shift is held
	float mouseSensitivity = 0.1f;

	Input input;
	GLFWwindow* window = nullptr;

private:
	void updateVectorsFromYawPitch();
	float lastMouseX = 0.0f;
	float lastMouseY = 0.0f;
	bool firstMouse = true;
};

