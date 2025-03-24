
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "mnist.h"
#include "mnist_data.h"
#include "predictive_maintenance_model.h"
#include "predictive_maintenance_model_data.h"

/* USER CODE BEGIN includes */

 extern UART_HandleTypeDef huart2;

#define BYTES_IN_FLOATS 5 * 4  // 5 caractéristiques d'entrée, chaque caractéristique est un float (4 octets)

 #define TIMEOUT 1000

 #define SYNCHRONISATION 0xAB

 #define ACKNOWLEDGE 0xCD

#define CLASS_NUMBER 6  // 6 classes pour les types de pannes (TWF, HDF, PWF, OSF, RNF, No Failure)


/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_PREDICTIVE_MAINTENANCE_MODEL_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_PREDICTIVE_MAINTENANCE_MODEL_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_PREDICTIVE_MAINTENANCE_MODEL_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_PREDICTIVE_MAINTENANCE_MODEL_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_PREDICTIVE_MAINTENANCE_MODEL_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_PREDICTIVE_MAINTENANCE_MODEL_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_PREDICTIVE_MAINTENANCE_MODEL_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_PREDICTIVE_MAINTENANCE_MODEL_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_PREDICTIVE_MAINTENANCE_MODEL_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};
ai_handle data_activations1[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle predictive_maintenance_model = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_predictive_maintenance_model_create_and_init(&predictive_maintenance_model, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_predictive_maintenance_model_create_and_init");
    return -1;
  }

  ai_input = ai_predictive_maintenance_model_inputs_get(predictive_maintenance_model, NULL);
  ai_output = ai_predictive_maintenance_model_outputs_get(predictive_maintenance_model, NULL);

#if defined(AI_PREDICTIVE_MAINTENANCE_MODEL_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_MODEL_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_MODEL_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_PREDICTIVE_MAINTENANCE_MODEL_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_MODEL_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_MODEL_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_predictive_maintenance_model_run(predictive_maintenance_model, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_predictive_maintenance_model_get_error(predictive_maintenance_model),
        "ai_predictive_maintenance_model_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8 *data[])
{
    unsigned char tmp[BYTES_IN_FLOATS] = {0};  // Taille des données d'entrée : 5 caractéristiques
    int num_elements = sizeof(tmp) / sizeof(tmp[0]);
    int num_floats = num_elements / 4;

    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t *)tmp, sizeof(tmp), TIMEOUT);
    if (status != HAL_OK)
    {
        printf("Failed to receive data from UART. Error code: %d\n", status);
        return (1);
    }

    if (num_elements % 4 != 0)
    {
        printf("The array length is not a multiple of 4 bytes. Cannot reconstruct floats.\n");
        return (1);
    }

    // Reconstruction des floats depuis les bytes reçus
    for (size_t i = 0; i < num_floats; i++)
    {
        unsigned char bytes[4] = {0};
        for (size_t j = 0; j < 4; j++)
        {
            bytes[j] = tmp[i * 4 + j];
        }
        ((uint8_t *)data)[(i * 4)] = bytes[0];
        ((uint8_t *)data)[(i * 4 + 1)] = bytes[1];
        ((uint8_t *)data)[(i * 4 + 2)] = bytes[2];
        ((uint8_t *)data)[(i * 4 + 3)] = bytes[3];
    }

    return (0);
}


int post_process(ai_i8 *data[])
{
    if (data == NULL)
    {
        printf("The output data is NULL.\n");
        return (1);
    }

    uint8_t *output = data;
    float outs[CLASS_NUMBER] = {0.0};
    uint8_t outs_uint8[CLASS_NUMBER] = {0};

    // Convertir la probabilité en float
    for (size_t i = 0; i < CLASS_NUMBER; i++)
    {
        uint8_t temp[4] = {0};
        // Extraire 4 octets pour reconstruire un float
        for (size_t j = 0; j < 4; j++)
        {
            temp[j] = output[i * 4 + j];
        }
        outs[i] = *(float *)&temp;
        outs_uint8[i] = (char)(outs[i] * 255);  // Convertir le float en uint8 pour UART
    }

    // Transmettre les résultats via UART
    HAL_StatusTypeDef status = HAL_UART_Transmit(&huart2, (uint8_t *)outs_uint8, sizeof(outs_uint8), TIMEOUT);
    if (status != HAL_OK)
    {
        printf("Failed to transmit data to UART. Error code: %d\n", status);
        return (1);
    }

    return 0;
}


void synchronize_UART(void)

{

    bool is_synced = 0;

    unsigned char rx[2] = {0};

    unsigned char tx[2] = {ACKNOWLEDGE, 0};

    while (!is_synced)

    {

      HAL_UART_Receive(&huart2, (uint8_t *)rx, sizeof(rx), TIMEOUT);

      if (rx[0] == SYNCHRONISATION)

      {

        HAL_UART_Transmit(&huart2, (uint8_t *)tx, sizeof(tx), TIMEOUT);

        is_synced = 1;

      }

    }

    return;

}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = -1;

  printf("TEMPLATE - run - main loop\r\n");

  if (predictive_maintenance_model) {

    do {
      /* 1 - acquire and pre-process input data */
      res = acquire_and_process_data(data_ins);
      /* 2 - process the data - call inference engine */
      if (res == 0)
        res = ai_run();
      /* 3- post-process the predictions */
      if (res == 0)
        res = post_process(data_outs);
    } while (res==0);
  }

  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
/* Multiple network support --------------------------------------------------*/

#include <string.h>
#include "ai_datatypes_defines.h"

static const ai_network_entry_t networks[AI_MNETWORK_NUMBER] =
{
    {
        .name = (const char *)AI_PREDICTIVE_MAINTENANCE_MODEL_MODEL_NAME,
        .config = AI_PREDICTIVE_MAINTENANCE_MODEL_DATA_CONFIG,
        .ai_get_report = ai_predictive_maintenance_model_get_report,
        .ai_create = ai_predictive_maintenance_model_create,
        .ai_destroy = ai_predictive_maintenance_model_destroy,
        .ai_get_error = ai_predictive_maintenance_model_get_error,
        .ai_init = ai_predictive_maintenance_model_init,
        .ai_run = ai_predictive_maintenance_model_run,
        .ai_forward = ai_predictive_maintenance_model_forward,
        .ai_data_params_get = ai_predictive_maintenance_model_data_params_get,
        .activations = data_activations1
    },
};

struct network_instance {
     const ai_network_entry_t *entry;
     ai_handle handle;
     ai_network_params params;
};

/* Number of instance is aligned on the number of network */
AI_STATIC struct network_instance gnetworks[AI_MNETWORK_NUMBER] = {0};

AI_DECLARE_STATIC
ai_bool ai_mnetwork_is_valid(const char* name,
        const ai_network_entry_t *entry)
{
    if (name && (strlen(entry->name) == strlen(name)) &&
            (strncmp(entry->name, name, strlen(entry->name)) == 0))
        return true;
    return false;
}

AI_DECLARE_STATIC
struct network_instance *ai_mnetwork_handle(struct network_instance *inst)
{
    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
        if ((inst) && (&gnetworks[i] == inst))
            return inst;
        else if ((!inst) && (gnetworks[i].entry == NULL))
            return &gnetworks[i];
    }
    return NULL;
}

AI_DECLARE_STATIC
void ai_mnetwork_release_handle(struct network_instance *inst)
{
    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
        if ((inst) && (&gnetworks[i] == inst)) {
            gnetworks[i].entry = NULL;
            return;
        }
    }
}

AI_API_ENTRY
const char* ai_mnetwork_find(const char *name, ai_int idx)
{
    const ai_network_entry_t *entry;

    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
        entry = &networks[i];
        if (ai_mnetwork_is_valid(name, entry))
            return entry->name;
        else {
            if (!idx--)
                return entry->name;
        }
    }
    return NULL;
}

AI_API_ENTRY
ai_error ai_mnetwork_create(const char *name, ai_handle* network,
        const ai_buffer* network_config)
{
    const ai_network_entry_t *entry;
    const ai_network_entry_t *found = NULL;
    ai_error err;
    struct network_instance *inst = ai_mnetwork_handle(NULL);

    if (!inst) {
        err.type = AI_ERROR_ALLOCATION_FAILED;
        err.code = AI_ERROR_CODE_NETWORK;
        return err;
    }

    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
        entry = &networks[i];
        if (ai_mnetwork_is_valid(name, entry)) {
            found = entry;
            break;
        }
    }

    if (!found) {
        err.type = AI_ERROR_INVALID_PARAM;
        err.code = AI_ERROR_CODE_NETWORK;
        return err;
    }

    if (network_config == NULL)
        err = found->ai_create(network, found->config);
    else
        err = found->ai_create(network, network_config);
    if ((err.code == AI_ERROR_CODE_NONE) && (err.type == AI_ERROR_NONE)) {
        inst->entry = found;
        inst->handle = *network;
        *network = (ai_handle*)inst;
    }

    return err;
}

AI_API_ENTRY
ai_handle ai_mnetwork_destroy(ai_handle network)
{
    struct network_instance *inn;
    inn =  ai_mnetwork_handle((struct network_instance *)network);
    if (inn) {
        ai_handle hdl = inn->entry->ai_destroy(inn->handle);
        if (hdl != inn->handle) {
            ai_mnetwork_release_handle(inn);
            network = AI_HANDLE_NULL;
        }
    }
    return network;
}

AI_API_ENTRY
ai_bool ai_mnetwork_get_report(ai_handle network, ai_network_report* report)
{
    struct network_instance *inn;
    inn =  ai_mnetwork_handle((struct network_instance *)network);
    if (inn)
        return inn->entry->ai_get_report(inn->handle, report);
    else
        return false;
}

AI_API_ENTRY
ai_error ai_mnetwork_get_error(ai_handle network)
{
    struct network_instance *inn;
    ai_error err;
    err.type = AI_ERROR_INVALID_PARAM;
    err.code = AI_ERROR_CODE_NETWORK;

    inn =  ai_mnetwork_handle((struct network_instance *)network);
    if (inn)
        return inn->entry->ai_get_error(inn->handle);
    else
        return err;
}

AI_API_ENTRY
ai_bool ai_mnetwork_init(ai_handle network)
{
    struct network_instance *inn;
    ai_network_params par;

    inn =  ai_mnetwork_handle((struct network_instance *)network);
    if (inn) {
        inn->entry->ai_data_params_get(&par);
        for (int idx=0; idx < par.map_activations.size; idx++)
          AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&par.map_activations, idx, inn->entry->activations[idx]);
        return inn->entry->ai_init(inn->handle, &par);
    }
    else
        return false;
}

AI_API_ENTRY
ai_i32 ai_mnetwork_run(ai_handle network, const ai_buffer* input,
        ai_buffer* output)
{
    struct network_instance* inn;
    inn =  ai_mnetwork_handle((struct network_instance *)network);
    if (inn)
        return inn->entry->ai_run(inn->handle, input, output);
    else
        return 0;
}

AI_API_ENTRY
ai_i32 ai_mnetwork_forward(ai_handle network, const ai_buffer* input)
{
    struct network_instance *inn;
    inn =  ai_mnetwork_handle((struct network_instance *)network);
    if (inn)
        return inn->entry->ai_forward(inn->handle, input);
    else
        return 0;
}

AI_API_ENTRY
 int ai_mnetwork_get_private_handle(ai_handle network,
         ai_handle *phandle,
         ai_network_params *pparams)
 {
     struct network_instance* inn;
     inn =  ai_mnetwork_handle((struct network_instance *)network);
     if (inn && phandle && pparams) {
         *phandle = inn->handle;
         *pparams = inn->params;
         return 0;
     }
     else
         return -1;
 }

#ifdef __cplusplus
}
#endif
