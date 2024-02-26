#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "opengl/stb_image.h"
#include "IdentifyGoalposts.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "opengl/shader_m.h"

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;


glm::vec3 postPositions[];

float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
};


glm::mat4 MoveCamera(glm::mat4 view, float roll, float pitch, float yaw, float xTranslation, float yTranslation, float zTranslation)
{
    float reverse = -1.0f;
    float smallNumber = 0.0000000000001f;
    

    

    
    //up down (pitch)
    if (pitch != 0)
    {
        view = glm::rotate(view, glm::radians(pitch), glm::vec3(-1.0f, 0.0f, 0.0f));
    }

    // rotate left/right (yaw)
    if (yaw != 0)
    {
        std::cout << std::to_string(yaw) << std::endl;
        view = glm::rotate(view, glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    // turn (roll)
    if (roll != 0)
    {
        view = glm::rotate(view, glm::radians((roll)), glm::vec3(0.0f, 0.0f, 1.0f));
    }



    view = glm::translate(view, glm::vec3(-(yTranslation + smallNumber), -(zTranslation +  smallNumber), -(xTranslation + smallNumber)));

    //view = glm::translate(view, glm::vec3(reverse * (xTranslation + smallNumber), reverse * (zTranslation + smallNumber), reverse * (yTranslation + smallNumber)));

    std::cout << std::to_string(yTranslation) + ", " + std::to_string(zTranslation) + ", " + std::to_string((xTranslation)) << std::endl;
    std::cout << std::to_string(pitch) + ", " + std::to_string(yaw) + ", " + std::to_string((roll)) << std::endl;

    return view;
}

Mat ConvertFrame()
{
    int height = 1080;
    int width = 1920;

    Mat img(height, width, CV_8UC3);
    Mat flipped;

    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);

    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());

    glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);

    cv::flip(img, flipped, 0);

    return flipped;
}

int RunGL()
{
    int counter = 0;

    //Init the camera data
    RunImageProcessing();

    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    //Hide the window offscreen
    //glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    Shader ourShader("6.3.coordinate_systems.vs", "6.3.coordinate_systems.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    //float yScale = 1;

    //float scale = 0.01f;
    float scale = 1.0f;

    // world space positions of our cubes
    /*glm::vec3 postPositions[] = {
        glm::vec3(5010.0f * scale, 475.0f, -870.0f * scale),
        glm::vec3(5010.0f * scale,  475.0f, 760.0f * scale),
        glm::vec3(5010.0f * scale,  762.5f, 210.0f * scale),
        glm::vec3(5010.0f * scale,  762.5f, -360.0f * scale),
        glm::vec3(-5010.0f * scale,  475.0f, 760.0f * scale),
        glm::vec3(-5010.0f * scale,  475.0f, -870.0f * scale),
        glm::vec3(-5010.0f * scale,  762.5f, -360.0f * scale),
        glm::vec3(-5010.0f * scale,  762.5f, 210.0f * scale)
    };
    */
    glm::vec3 postPositions[] = {
        glm::vec3(-870.0f * scale, 475.0f, 5010.0f * scale),
        glm::vec3(760.0f * scale,  475.0f, 5010.0f * scale),
        glm::vec3(210.0f * scale,  762.5f, 5010.0f * scale),
        glm::vec3(-360.0f * scale,  762.5f, 5010.0f * scale),
        glm::vec3(760.0f * scale,  475.0f, -5010.0f * scale),
        glm::vec3(-870.0f * scale,  475.0f, -5010.0f * scale),
        glm::vec3(-360.0f * scale,  762.5f, -5010.0f * scale),
        glm::vec3(210.0f * scale,  762.5f, -5010.0f * scale)
    };

    glm::vec3 markerPositions[] = {
    glm::vec3(-3500.0f * scale, 0.0f, 5010.0f * scale),
    glm::vec3(3500.0f * scale,  0.0f, 5010.0f * scale),
    glm::vec3(-3500.0f * scale,  0.0f, -5010.0f * scale),
    glm::vec3(3500.0f * scale,  0.0f, -5010.0f * scale),

    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        processInput(window);

        // render
        //background colour for window
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

        // activate shader
        ourShader.use();

        //Get next set of camera values
        IterateNextFrame();

        // create transformations
        glm::mat4 view = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        glm::mat4 projection = glm::mat4(1.0f);
        //vertical field of view = 2 atan(0.5 height / focallength)
        //https://sdk-forum.dji.net/hc/en-us/articles/11317874071065-How-to-calculate-the-FoV-of-the-camera-lens
        float fov = 60.0f;
        float nearDistance = 0.1f;
        float farDistance = 20000.0f;
        projection = glm::perspective(glm::radians(fov), (float)SCR_WIDTH / (float)SCR_HEIGHT, nearDistance, farDistance);

        view = MoveCamera(view, GetCameraRoll(), GetCameraPitch(), GetCameraYaw(), GetCameraXPosition(), GetCameraYPosition(), GetCameraZPosition());


        // pass transformation matrices to the shader
        ourShader.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        ourShader.setMat4("view", view);

        // render boxes
        glBindVertexArray(VAO);
        for (unsigned int i = 0; i < 13; i++)
        {
            // calculate the model matrix for each object and pass it to shader before drawing
            glm::mat4 model = glm::mat4(1.0f);


            if (i == 0 || i == 1 || i == 4 || i == 5)
            {
                model = glm::translate(model, postPositions[i]);
                model = glm::scale(model, glm::vec3(50.0f * scale, 950.0f * scale, 50.0f * scale));
            }
            else if (i == 2 || i == 3 || i == 6 || i == 7)
            {
                model = glm::translate(model, postPositions[i]);
                model = glm::scale(model, glm::vec3(50.0f * scale, 1525.0f * scale, 50.0f * scale));
            }

            else if (i == 8)
            {
                model = glm::scale(model, glm::vec3(50.0f * scale, 50.0f * scale, 50.0f * scale));
            }
            else
            {
                model = glm::translate(model, markerPositions[i-9]);
                model = glm::scale(model, glm::vec3(50.0f * scale, 50.0f * scale, 50.0f * scale));
            }
            
            ourShader.setMat4("model", model);

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        //glfwPollEvents();

        //Convert to Mat 
        cv::Mat m = ConvertFrame();

        //Send to be processed (functions now move to identifygoalposts.cpp)
        ProcessGLFrame(m, counter);
        
        counter++;
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}



int main()
{
    RunGL();
    //RunImageProcessing();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}