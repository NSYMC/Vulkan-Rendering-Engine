#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include "Input.h"

class Camera
{
public:
	Camera();
	glm::mat4 GetViewMatrix();
	void Update(float deltaTime); 
	~Camera();
	glm::vec3 position = glm::vec3(5.0f,2.0f,5.0f);
	glm::vec3 front = glm::vec3(0,0,-1);
	glm::vec3 up = glm::vec3(0,1,0);
	glm::vec3 right = glm::vec3(1,0,0);
	float yaw;
	float pitch;
	float movementSpeed = 10;
	float mouseSens;
	Input input;
};

