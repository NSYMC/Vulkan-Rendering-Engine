#include "ModelPanel.h"
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <windows.h>
#include <algorithm>
#include <stdexcept>

namespace {
	std::vector<std::string> getObjPathsInFolder(const std::string& folderPath) {
		std::vector<std::string> paths;
		std::string searchPath = folderPath;
		if (searchPath.back() != '/' && searchPath.back() != '\\')
			searchPath += "\\";
		searchPath += "*.obj";
		WIN32_FIND_DATAA findData = {};
		HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
		if (hFind == INVALID_HANDLE_VALUE)
			return paths;
		do {
			if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				paths.push_back(folderPath + "/" + std::string(findData.cFileName));
		} while (FindNextFileA(hFind, &findData));
		FindClose(hFind);
		std::sort(paths.begin(), paths.end());
		return paths;
	}
}

ModelPanel* g_modelPanelInstance = nullptr;

LRESULT CALLBACK ModelPanel::PanelWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	ModelPanel* self = g_modelPanelInstance;
	if (!self || self->hwnd_ != hwnd)
		return DefWindowProcA(hwnd, msg, wParam, lParam);

	switch (msg) {
	case WM_COMMAND:
		if (LOWORD(wParam) == 1001 && HIWORD(wParam) == LBN_DBLCLK) {
			int idx = (int)SendMessageA(self->listbox_, LB_GETCURSEL, 0, 0);
			if (idx >= 0) self->onDoubleClick(idx);
		}
		break;
	case WM_LBUTTONDOWN: {
		POINT pt = { LONG(lParam) & 0xFFFF, LONG(lParam) >> 16 };
		ClientToScreen(hwnd, &pt);
		POINT listPt = pt;
		ScreenToClient(self->listbox_, &listPt);
		DWORD res = (DWORD)SendMessageA(self->listbox_, LB_ITEMFROMPOINT, 0, MAKELPARAM(listPt.x, listPt.y));
		int idx = (int)LOWORD(res);
		if (idx >= 0 && idx < (int)self->modelPaths_.size()) {
			self->dragIndex_ = idx;
			SetCapture(hwnd);
		}
		break;
	}
	case WM_LBUTTONUP:
		if (self->dragIndex_ >= 0) {
			ReleaseCapture();
			POINT pt;
			GetCursorPos(&pt);
			self->onDragEnd(self->dragIndex_, pt.x, pt.y);
			self->dragIndex_ = -1;
		}
		break;
	case WM_DESTROY:
		self->hwnd_ = nullptr;
		self->listbox_ = nullptr;
		break;
	default:
		return DefWindowProcA(hwnd, msg, wParam, lParam);
	}
	return 0;
}

ModelPanel::ModelPanel() {
	g_modelPanelInstance = this;
}

ModelPanel::~ModelPanel() {
	destroy();
	g_modelPanelInstance = nullptr;
}

void ModelPanel::create(void* mainWindowHWND, int mainWidth, int mainHeight, OnAddModelFn addCallback) {
	if (hwnd_) return;
	mainWindowHWND_ = (HWND)mainWindowHWND;
	onAddModel_ = std::move(addCallback);
	modelPaths_ = getObjPathsInFolder("Models");

	WNDCLASSEXA wc = {};
	wc.cbSize = sizeof(wc);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = PanelWndProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszClassName = "ModelPanelClass";
	if (!RegisterClassExA(&wc))
		throw std::runtime_error("ModelPanel: RegisterClassEx failed");

	HMONITOR hMon = MonitorFromWindow(mainWindowHWND_, MONITOR_DEFAULTTONEAREST);
	MONITORINFO mi = {};
	mi.cbSize = sizeof(mi);
	GetMonitorInfoA(hMon, &mi);
	workArea_ = mi.rcWork;
	int workW = workArea_.right - workArea_.left;
	int workH = workArea_.bottom - workArea_.top;
	int panelH = panelHeight_;
	int contentH = workH - panelH;
	if (contentH < 300) contentH = workH / 2;

	int x = workArea_.left;
	int y = workArea_.top + contentH;
	int w = workW;
	int h = panelH;

	// Pencereyi olustur, baslangicta gizli (M ile acilir)
	HWND hPanel = CreateWindowExA(0, "ModelPanelClass", "Modeller - surukle veya cift tikla (M: kapat)",
		WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
		x, y, w, h, nullptr, nullptr, wc.hInstance, nullptr);
	if (!hPanel)
		throw std::runtime_error("ModelPanel: CreateWindow failed");
	hwnd_ = hPanel;
	visible_ = false;
	ShowWindow(hPanel, SW_HIDE);

	HWND hList = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", nullptr,
		WS_CHILD | WS_VISIBLE | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT | WS_VSCROLL,
		10, 35, w - 24, h - 50, hPanel, (HMENU)1001, wc.hInstance, nullptr);
	if (!hList)
		throw std::runtime_error("ModelPanel: ListBox failed");
	listbox_ = hList;

	CreateWindowExA(0, "STATIC", "Modelleri sahneye surukleyin veya liste uzerinde cift tiklayin.",
		WS_CHILD | WS_VISIBLE, 10, 8, (int)(w - 20), 20, hPanel, nullptr, wc.hInstance, nullptr);

	for (const auto& path : modelPaths_) {
		size_t slash = path.find_last_of("/\\");
		std::string name = slash != std::string::npos ? path.substr(slash + 1) : path;
		SendMessageA(listbox_, LB_ADDSTRING, 0, (LPARAM)name.c_str());
	}
}

void ModelPanel::destroy() {
	if (hwnd_) {
		DestroyWindow(hwnd_);
		hwnd_ = nullptr;
		listbox_ = nullptr;
		mainWindowHWND_ = nullptr;
	}
	UnregisterClassA("ModelPanelClass", GetModuleHandle(nullptr));
}

void ModelPanel::refreshModelList() {
	if (!listbox_) return;
	SendMessageA(listbox_, LB_RESETCONTENT, 0, 0);
	modelPaths_ = getObjPathsInFolder("Models");
	for (const auto& path : modelPaths_) {
		size_t slash = path.find_last_of("/\\");
		std::string name = slash != std::string::npos ? path.substr(slash + 1) : path;
		SendMessageA(listbox_, LB_ADDSTRING, 0, (LPARAM)name.c_str());
	}
}

void ModelPanel::setPositionBelow(int mainX, int mainY, int mainWidth, int mainHeight) {
	if (!hwnd_) return;
	SetWindowPos(hwnd_, nullptr, mainX, mainY + mainHeight, mainWidth, panelHeight_, SWP_NOZORDER | SWP_NOACTIVATE);
}

void ModelPanel::updateLayout(bool panelVisible) {
	if (!hwnd_ || !mainWindowHWND_) return;
	int workW = workArea_.right - workArea_.left;
	int workH = workArea_.bottom - workArea_.top;
	int panelH = panelHeight_;

	// Ana pencereyi boyutlandirmiyoruz (Vulkan swapchain bozulmasin).
	// Panel sadece altta gosterilir/gizlenir, acikken 3D penceresinin altini kaplar.
	if (panelVisible) {
		SetWindowPos(hwnd_, nullptr, workArea_.left, workArea_.top + workH - panelH, workW, panelH, SWP_NOZORDER);
		ShowWindow(hwnd_, SW_SHOW);
		SetWindowPos(hwnd_, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	} else {
		ShowWindow(hwnd_, SW_HIDE);
	}
}

void ModelPanel::toggleVisibility() {
	if (!hwnd_) return;
	visible_ = !visible_;
	updateLayout(visible_);
}

void ModelPanel::onDoubleClick(int listIndex) {
	if (listIndex < 0 || listIndex >= (int)modelPaths_.size() || !onAddModel_) return;
	onAddModel_(modelPaths_[listIndex]);
}

void ModelPanel::onDragEnd(int listIndex, int screenX, int screenY) {
	if (listIndex < 0 || listIndex >= (int)modelPaths_.size() || !onAddModel_ || !mainWindowHWND_) return;
	POINT pt = { screenX, screenY };
	HWND hOver = WindowFromPoint(pt);
	if (hOver == mainWindowHWND_ || IsChild(mainWindowHWND_, hOver))
		onAddModel_(modelPaths_[listIndex]);
}

void ModelPanel::pollMessages() {
	MSG msg;
	while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessageA(&msg);
	}
}
