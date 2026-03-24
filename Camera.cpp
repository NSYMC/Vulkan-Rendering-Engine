#include "Camera.h"
#include <GLFW/glfw3.h>

Camera::Camera()
{
	updateVectorsFromYawPitch();
}

glm::mat4 Camera::GetViewMatrix()
{
	return glm::lookAt(position, position + front, up);
}

void Camera::updateVectorsFromYawPitch()
{
	float yawRad = glm::radians(yaw);
	float pitchRad = glm::radians(pitch);
	// yaw 0 = -Z, yaw 90 = +X (right-handed, Y up)
	front.x = cos(pitchRad) * sin(yawRad);
	front.y = sin(pitchRad);
	front.z = -cos(pitchRad) * cos(yawRad);
	front = glm::normalize(front);
	right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
	up = glm::normalize(glm::cross(right, front));
}

void Camera::Update(float deltaTime)
{
	// Mouse look only while right mouse button is held (Unity-style)
	const bool rotateWithMouse = input.GetMouseButton(GLFW_MOUSE_BUTTON_RIGHT);
	if (window && rotateWithMouse) {
		double mouseX, mouseY;
		glfwGetCursorPos(window, &mouseX, &mouseY);

		if (firstMouse) {
			lastMouseX = static_cast<float>(mouseX);
			lastMouseY = static_cast<float>(mouseY);
			firstMouse = false;
		}

		float deltaX = static_cast<float>(mouseX - lastMouseX);
		float deltaY = static_cast<float>(lastMouseY - mouseY); // Y flipped: up = positive

		lastMouseX = static_cast<float>(mouseX);
		lastMouseY = static_cast<float>(mouseY);

		yaw += mouseSensitivity * deltaX;
		pitch += mouseSensitivity * deltaY;
		if (pitch > 89.0f) pitch = 89.0f;
		if (pitch < -89.0f) pitch = -89.0f;

		updateVectorsFromYawPitch();

		// Warp cursor to center for unlimited rotation (FPS-style)
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		double centerX = width / 2.0;
		double centerY = height / 2.0;
		glfwSetCursorPos(window, centerX, centerY);
		lastMouseX = static_cast<float>(centerX);
		lastMouseY = static_cast<float>(centerY);
		// Sync Input so next frame sees center ù avoids repeated delta and continuous turn
	} else {
		firstMouse = true; // next time user holds right button we don't get a jump
	}

	// Movement speed: higher when Shift is held
	float speed = movementSpeed * deltaTime;
	if (input.GetKey(GLFW_KEY_LEFT_SHIFT) || input.GetKey(GLFW_KEY_RIGHT_SHIFT))
		speed *= sprintMultiplier;

	if (input.GetKey(GLFW_KEY_W))
		position += front * speed;
	if (input.GetKey(GLFW_KEY_S))
		position -= front * speed;
	if (input.GetKey(GLFW_KEY_A))
		position -= right * speed;
	if (input.GetKey(GLFW_KEY_D))
		position += right * speed;
}

Camera::~Camera()
{
}
