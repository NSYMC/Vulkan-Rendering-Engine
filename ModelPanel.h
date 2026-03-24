#pragma once

#include <string>
#include <vector>
#include <functional>
#include <windows.h>

// Unity tarzı: altta model listesi, sürükle-bırak veya çift tıkla sahneye ekleme.
// Ana pencere (GLFW) altında bir Win32 panel penceresi açar.
class ModelPanel {
public:
	using OnAddModelFn = std::function<void(const std::string& modelPath)>;

	ModelPanel();
	~ModelPanel();

	// Ana pencere ve drop hedefi için HWND (GLFW: glfwGetWin32Window(window))
	// addCallback: model path verilir, sahneye ekleme burada yapılır
	void create(void* mainWindowHWND, int mainWidth, int mainHeight, OnAddModelFn addCallback);
	void destroy();

	// Models klasöründeki .obj listesini tara ve listeyi güncelle
	void refreshModelList();

	// Pencere konumunu güncelle (ana pencere taşınınca paneli de taşı)
	void setPositionBelow(int mainX, int mainY, int mainWidth, int mainHeight);

	// M tusuyla: menuyu ac/kapat (altta goster/gizle)
	void toggleVisibility();

	// Windows mesajlarını işle (main loop'tan çağır)
	static void pollMessages();

private:
	HWND hwnd_ = nullptr;
	HWND listbox_ = nullptr;
	HWND mainWindowHWND_ = nullptr;
	std::vector<std::string> modelPaths_;
	OnAddModelFn onAddModel_;
	int dragIndex_ = -1;
	bool visible_ = false;
	RECT workArea_ = {};
	static const int panelHeight_ = 220;

	void updateLayout(bool panelVisible);

	static LRESULT CALLBACK PanelWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
	void onDoubleClick(int listIndex);
	void onDragEnd(int listIndex, int screenX, int screenY);
};
