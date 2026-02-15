#include "Camera.h"



Camera::Camera()
{
}

glm::mat4 Camera::GetViewMatrix()
{
	return glm::lookAt(position, position+front, up);
}

void Camera::Update(float deltaTime)
{
	if (input.GetKey(GLFW_KEY_W)) {
		position += front * movementSpeed * deltaTime;
		std::cout << "calisti" << std::endl;
	}
	if (input.GetKey(GLFW_KEY_S)) {
		position -= front * movementSpeed * deltaTime;
	}
	if (input.GetKey(GLFW_KEY_A)) {
		position -= right * movementSpeed * deltaTime;
	}
	if (input.GetKey(GLFW_KEY_D)) {
		position += right * movementSpeed * deltaTime;
	}


}

Camera::~Camera()
{
}


/*
W(Ýleri) : Position += Front * speed * deltaTime
S(Geri) : Position -= Front * speed * deltaTime
A(Sola) : Position -= Right * speed * deltaTime(Sað vektörünün tersi).
D(Saða) : Position += Right * speed * deltaTime
*/
