#include <iostream>
#include <cstdlib>
#include <hdx/app_state.h>
#include "Application.h"

AppState app_state;

Application *app;

float current_time, delta_time;

int main(int argc, char *args[])
{
    app = new Application();

    while (app_state.running)
    {
        app->window->loop(app_state, delta_time);

        if(!app_state.paused)
            app->update(delta_time, app_state);
    }

    delete app;

    return 0;
}