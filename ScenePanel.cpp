#include "ScenePanel.h"
#include "Scene.h"
#include "Utilities.h"
#include <windows.h>
#include <cstdio>
#include <stdexcept>

ScenePanel* g_scenePanelInstance = nullptr;

LRESULT CALLBACK ScenePanel::PanelWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	ScenePanel* self = g_scenePanelInstance;
	if (!self || self->hwnd_ != hwnd)
		return DefWindowProcA(hwnd, msg, wParam, lParam);

	switch (msg) {
	case WM_COMMAND: {
		WORD id = LOWORD(wParam);
		if (id == 1002) {
			int sel = (int)SendMessageA(self->listbox_, LB_GETCURSEL, 0, 0);
			if (sel >= 0) self->onRemoveClicked(sel);
		}
		break;
	}
	case WM_DESTROY:
		self->hwnd_ = nullptr;
		self->listbox_ = nullptr;
		self->scene_ = nullptr;
		break;
	default:
		return DefWindowProcA(hwnd, msg, wParam, lParam);
	}
	return 0;
}

ScenePanel::ScenePanel() {
	g_scenePanelInstance = this;
}

ScenePanel::~ScenePanel() {
	destroy();
	g_scenePanelInstance = nullptr;
}

void ScenePanel::create(void* mainWindowHWND, int mainWidth, int mainHeight, Scene* scene, OnRemoveCallback onRemove) {
	if (hwnd_) return;
	mainWindowHWND_ = (HWND)mainWindowHWND;
	scene_ = scene;
	onRemove_ = std::move(onRemove);

	WNDCLASSEXA wc = {};
	wc.cbSize = sizeof(wc);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = PanelWndProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
	wc.lpszClassName = "ScenePanelClass";
	if (!RegisterClassExA(&wc))
		throw std::runtime_error("ScenePanel: RegisterClassEx failed");

	HMONITOR hMon = MonitorFromWindow(mainWindowHWND_, MONITOR_DEFAULTTONEAREST);
	MONITORINFO mi = {};
	mi.cbSize = sizeof(mi);
	GetMonitorInfoA(hMon, &mi);
	workArea_ = mi.rcWork;
	int w = panelWidth_;
	int h = workArea_.bottom - workArea_.top;

	hwnd_ = CreateWindowExA(0, "ScenePanelClass", "Sahne (O: kapat/ac)",
		WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
		workArea_.left, workArea_.top, w, h, nullptr, nullptr, wc.hInstance, nullptr);
	if (!hwnd_)
		throw std::runtime_error("ScenePanel: CreateWindow failed");
	visible_ = false;
	ShowWindow(hwnd_, SW_HIDE);

	CreateWindowExA(0, "STATIC", "Tum objeler ve isiklar. Secip Kaldir ile silin.",
		WS_CHILD | WS_VISIBLE, 10, 8, w - 60, 18, hwnd_, nullptr, wc.hInstance, nullptr);

	listbox_ = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", nullptr,
		WS_CHILD | WS_VISIBLE | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT | WS_VSCROLL,
		10, 32, w - 60, h - 90, hwnd_, (HMENU)1001, wc.hInstance, nullptr);
	if (!listbox_)
		throw std::runtime_error("ScenePanel: ListBox failed");

	CreateWindowExA(0, "BUTTON", "Kaldir", WS_CHILD | WS_VISIBLE,
		10, h - 52, 80, 24, hwnd_, (HMENU)1002, wc.hInstance, nullptr);

	refreshList();
}

void ScenePanel::destroy() {
	if (hwnd_) {
		DestroyWindow(hwnd_);
		hwnd_ = nullptr;
		listbox_ = nullptr;
		mainWindowHWND_ = nullptr;
		scene_ = nullptr;
		selectedObjectIndex_ = nullptr;
		onRemove_ = nullptr;
	}
	UnregisterClassA("ScenePanelClass", GetModuleHandle(nullptr));
}

void ScenePanel::refreshList() {
	if (!listbox_ || !scene_) return;
	SendMessageA(listbox_, LB_RESETCONTENT, 0, 0);
	size_t objCount = scene_->GetObjectCount();
	size_t lightCount = scene_->getLightCount();
	for (size_t i = 0; i < objCount; i++) {
		char buf[48];
		snprintf(buf, sizeof(buf), "Object %zu (Mesh)", i);
		SendMessageA(listbox_, LB_ADDSTRING, 0, (LPARAM)buf);
	}
	for (size_t i = 0; i < lightCount; i++) {
		char buf[48];
		snprintf(buf, sizeof(buf), "Point Light %zu", i);
		SendMessageA(listbox_, LB_ADDSTRING, 0, (LPARAM)buf);
	}
	size_t spotCount = scene_->getSpotLightCount();
	for (size_t i = 0; i < spotCount; i++) {
		char buf[48];
		snprintf(buf, sizeof(buf), "Spot Light %zu", i);
		SendMessageA(listbox_, LB_ADDSTRING, 0, (LPARAM)buf);
	}
}

void ScenePanel::onRemoveClicked(int listIndex) {
	if (!scene_) return;
	size_t objCount = scene_->GetObjectCount();
	size_t lightCount = scene_->getLightCount();
	size_t spotCount = scene_->getSpotLightCount();
	int total = (int)(objCount + lightCount + spotCount);
	if (listIndex < 0 || listIndex >= total) return;

	if (listIndex < (int)objCount) {
		if (selectedObjectIndex_) {
			int& sel = *selectedObjectIndex_;
			if (sel == listIndex) sel = -1;
			else if (sel > listIndex) sel--;
		}
		scene_->RemoveObjectAt((size_t)listIndex);
	} else if (listIndex < (int)(objCount + lightCount)) {
		scene_->removeLight((size_t)(listIndex - (int)objCount));
		if (onRemove_) onRemove_(-1);
	} else {
		scene_->removeSpotLight((size_t)(listIndex - (int)objCount - (int)lightCount));
		if (onRemove_) onRemove_(-1);
	}
	refreshList();
}

void ScenePanel::setPositionLeft(int mainX, int mainY, int mainWidth, int mainHeight) {
	(void)mainX;
	(void)mainY;
	(void)mainWidth;
	(void)mainHeight;
	if (!hwnd_) return;
	int w = panelWidth_;
	int h = workArea_.bottom - workArea_.top;
	SetWindowPos(hwnd_, nullptr, workArea_.left, workArea_.top, w, h, SWP_NOZORDER | SWP_NOACTIVATE);
}

void ScenePanel::updateLayout(bool panelVisible) {
	if (!hwnd_ || !mainWindowHWND_) return;
	int w = panelWidth_;
	int h = workArea_.bottom - workArea_.top;
	if (panelVisible) {
		SetWindowPos(hwnd_, nullptr, workArea_.left, workArea_.top, w, h, SWP_NOZORDER);
		ShowWindow(hwnd_, SW_SHOW);
		SetWindowPos(hwnd_, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
		if (listbox_)
			SetWindowPos(listbox_, nullptr, 0, 0, w - 60, h - 90, SWP_NOMOVE | SWP_NOZORDER);
	} else
		ShowWindow(hwnd_, SW_HIDE);
}

void ScenePanel::toggleVisibility() {
	if (!hwnd_) return;
	visible_ = !visible_;
	updateLayout(visible_);
}

void ScenePanel::setSelectionAfterRemove(int* selectedObjectIndex) {
	selectedObjectIndex_ = selectedObjectIndex;
}

void ScenePanel::pollMessages() {
	MSG msg;
	while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessageA(&msg);
	}
}
