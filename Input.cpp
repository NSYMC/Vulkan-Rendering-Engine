#include "Input.h"

bool Input::keys[1024] = { false };
bool Input::lastKeys[1024] = { false };

Input::Input()
{

}

void Input::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key >= 0 && key < 1024) {
		if (action == GLFW_PRESS) {
			keys[key] = true;
			
		}
		else if (action == GLFW_RELEASE) {
			keys[key] = false;
		}
	}
}

void Input::cursor_positin_callback(GLFWwindow* window, double xpos, double ypos)
{
	mouseX = xpos;
	mouseY = ypos;
}

void Input::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button >= 0 && button <= 10) {
		if (action == GLFW_PRESS) {
			buttons[button] = true;
		}
		else if (action == GLFW_RELEASE) {
			buttons[button] = false;
		}
	}
}

bool Input::GetKey(int key)
{
	if (keys[key] == true) {
		return true;
	}
	return false;
}

bool Input::GetKeyDown(int key)
{
	if (keys[key] == true && lastKeys[key] == false)
		return true;
	return false;
}

bool Input::GetKeyUp(int key)
{
	if (keys[key] == false && lastKeys[key] == true)
		return true;
	return false;
}

void Input::Update()
{

	std::memcpy(lastKeys, keys, sizeof(keys));
	std::memcpy(lastButtons, buttons, sizeof(buttons));
}

Input::~Input()
{
}






