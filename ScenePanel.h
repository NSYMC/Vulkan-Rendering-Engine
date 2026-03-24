#pragma once

#include <windows.h>
#include <functional>

class Scene;

// Sol tarafta acilan panel: sahnedeki tum objeler (mesh + isik) listesi, kaldirilabilir.
class ScenePanel {
public:
	/** removedModelId = kaldirilan sahne objesinin modelId'si; isik kaldirildiysa -1. */
	using OnRemoveCallback = std::function<void(int removedModelId)>;

	ScenePanel();
	~ScenePanel();

	void create(void* mainWindowHWND, int mainWidth, int mainHeight, Scene* scene, OnRemoveCallback onRemove = nullptr);
	void destroy();

	void setPositionLeft(int mainX, int mainY, int mainWidth, int mainHeight);
	void toggleVisibility();
	void refreshList();  // sahne degisti sonra cagir
	static void pollMessages();

	// Secili obje indexi degisti (sol listeden kaldirma sonrasi main loop gunceller)
	void setSelectionAfterRemove(int* selectedObjectIndex);

private:
	HWND hwnd_ = nullptr;
	HWND listbox_ = nullptr;
	HWND mainWindowHWND_ = nullptr;
	Scene* scene_ = nullptr;
	int* selectedObjectIndex_ = nullptr;
	OnRemoveCallback onRemove_;
	bool visible_ = false;
	RECT workArea_ = {};
	static const int panelWidth_ = 260;

	void updateLayout(bool panelVisible);
	static LRESULT CALLBACK PanelWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
	void onRemoveClicked(int listIndex);
};
