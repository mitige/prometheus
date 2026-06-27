#include "GuiApp.h"

#include <cstdio>

int main(int, char**) {
	prometheus::GuiApp app;
	if (!app.Init(1440, 900)) {
		std::fprintf(stderr, "Failed to initialize Prometheus GUI\n");
		return 1;
	}

	app.Run();
	app.Shutdown();
	return 0;
}
