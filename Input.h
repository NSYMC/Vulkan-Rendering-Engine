#pragma once

#include <GLFW/glfw3.h>
#include <set>
#include <array>
#include <iostream>



class Input
{
public:

	Input();
	GLFWwindow* window;
	static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void cursor_positin_callback(GLFWwindow* window, double xpos, double ypos);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static bool GetKey(int key);
	static bool GetKeyDown(int key);
	static bool GetKeyUp(int key);
	static void Update();
	static double mouseX;
	static double mouseY;
	~Input();

private:
	static bool keys[1024];
	static bool lastKeys[1024];
	static bool buttons[10];
	static bool lastButtons[10];
};

