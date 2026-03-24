#pragma once

#include <windows.h>
#include <functional>

struct UserPointLight;
struct UserSpotLight;
class Scene;

// Isik paneli (L: ac/kapat): Point ve Spot isik ekle/duzenle. Sol panelden kaldirilabilir.
class LightPanel {
public:
	LightPanel();
	~LightPanel();

	using OnLightListChanged = std::function<void()>;
	void create(void* mainWindowHWND, int mainWidth, int mainHeight, Scene* scene, OnLightListChanged onLightListChanged = nullptr);
	void destroy();

	void setPositionBelow(int mainX, int mainY, int mainWidth, int mainHeight);
	void toggleVisibility();
	void refreshList();  // sahne isik sayisi degisti (sol panelden silme vs) sonra cagir
	static void pollMessages();

private:
	HWND hwnd_ = nullptr;
	HWND mainWindowHWND_ = nullptr;
	HWND btnAddPoint_ = nullptr;
	HWND btnAddSpot_ = nullptr;
	Scene* scene_ = nullptr;
	OnLightListChanged onLightListChanged_;
	int selectedIndex_ = 0;
	int selectedSpotIndex_ = 0;
	bool visible_ = false;
	RECT workArea_ = {};
	static const int panelHeight_ = 420;

	enum { IDC_COMBO = 2000, IDC_ADD_POINT, IDC_ENABLED, IDC_PX, IDC_PY, IDC_PZ, IDC_CR, IDC_CG, IDC_CB, IDC_INT, IDC_RANGE,
	      IDC_COMBO_SPOT, IDC_ADD_SPOT, IDC_ENABLED_SPOT, IDC_SPX, IDC_SPY, IDC_SPZ, IDC_SDX, IDC_SDY, IDC_SDZ,
	      IDC_SCR, IDC_SCG, IDC_SCB, IDC_SINT, IDC_SRANGE, IDC_SINNER, IDC_SOUTER };
	HWND combo_ = nullptr;
	HWND chkEnabled_ = nullptr;
	HWND editPX_ = nullptr, editPY_ = nullptr, editPZ_ = nullptr;
	HWND editCR_ = nullptr, editCG_ = nullptr, editCB_ = nullptr;
	HWND editInt_ = nullptr, editRange_ = nullptr;
	HWND comboSpot_ = nullptr;
	HWND chkEnabledSpot_ = nullptr;
	HWND editSpotPX_ = nullptr, editSpotPY_ = nullptr, editSpotPZ_ = nullptr;
	HWND editSpotDX_ = nullptr, editSpotDY_ = nullptr, editSpotDZ_ = nullptr;
	HWND editSpotCR_ = nullptr, editSpotCG_ = nullptr, editSpotCB_ = nullptr;
	HWND editSpotInt_ = nullptr, editSpotRange_ = nullptr, editSpotInner_ = nullptr, editSpotOuter_ = nullptr;

	void updateLayout(bool panelVisible);
	void refreshComboFromScene();
	void refreshSpotComboFromScene();
	void refreshEditControls();
	void refreshSpotEditControls();
	void applyFromEditControls();
	void applyFromSpotEditControls();
	static LRESULT CALLBACK PanelWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
};
