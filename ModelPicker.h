#pragma once

#include <string>
#include <vector>

struct GLFWwindow;
class VulkanRender;
class Scene;
class Input;

// Returns paths like "Models/mcl35m_2.obj" for each .obj in the given folder.
std::vector<std::string> getObjPathsInFolder(const std::string& folderPath);

// Simple UI: M toggles menu (window title), keys 1-9 add the corresponding model to the scene.
class ModelPickerUI {
public:
	ModelPickerUI(VulkanRender* renderer, Scene* scene, GLFWwindow* window);
	// Rescan Models folder (call once at startup or when you add new .obj files).
	void refreshModelList();
	// Call each frame; handles M (toggle menu) and 1-9 (add model).
	void update(Input& input);

	const std::vector<std::string>& getModelPaths() const { return modelPaths_; }
	bool isMenuVisible() const { return menuVisible_; }

private:
	VulkanRender* renderer_ = nullptr;
	Scene* scene_ = nullptr;
	GLFWwindow* window_ = nullptr;
	std::vector<std::string> modelPaths_;
	bool menuVisible_ = false;
	int addedCount_ = 0;  // for offsetting each new model so they don't stack
	std::string defaultTitle_;
	void updateWindowTitle();
};
