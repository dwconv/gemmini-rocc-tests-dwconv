#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// #define SKIP_CPU

//Now We are using this setting
#define IN_DIM 14
#define CHANNELS 32
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1



#define BATCH_SIZE 1

#define NO_BIAS false // false -> bias exist

#define OUT_DIM         ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)

#define ONE_BATCH_SIZE  (OUT_DIM*OUT_DIM*CHANNELS) 
#define ONE_ROW_SIZE    (OUT_DIM*CHANNELS)
#define ONE_COL_SIZE    (CHANNELS)

// #define ONE_BATCH_SIZE (3*2*2) 
// #define ONE_ROW_SIZE (2*2)
// #define ONE_COL_SIZE (2)

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i]){
            printf("MISMATCH POINT...!\r\n");
            printf("BATCH : %d, ",i/(ONE_BATCH_SIZE));
            int ONE_BATCH_IN_INDEX = i - (i/ONE_BATCH_SIZE) * ONE_BATCH_SIZE; 

            printf("ROW : %d, ",ONE_BATCH_IN_INDEX/ONE_ROW_SIZE);
            int ONE_ROW_IN_INDEX = ONE_BATCH_IN_INDEX - (ONE_BATCH_IN_INDEX/ONE_ROW_SIZE) * ONE_ROW_SIZE;

            printf("COL : %d, ",ONE_ROW_IN_INDEX/ONE_COL_SIZE);
            int ONE_COl_IN_INDEX = ONE_ROW_IN_INDEX - (ONE_ROW_IN_INDEX/ONE_COL_SIZE) * ONE_COL_SIZE;

            printf("CHANNEL : %d\r\n",ONE_COl_IN_INDEX);

            printf("------ MISMATCH MOMENT ------\r\n");
            printf("CPU..\r\n");
           
            for(int k_=0;k_<OUT_DIM;k_++){
                for(int l_=0;l_<OUT_DIM;l_++){
                    printf("%d\t",*(a + ((i/(ONE_BATCH_SIZE))*OUT_DIM*OUT_DIM*CHANNELS + k_*OUT_DIM*CHANNELS + l_*CHANNELS+(ONE_COl_IN_INDEX))));
                }
                printf("\n");
            }
            printf("\n");
        
            printf("GEMMINI..\r\n");

            for(int k_=0;k_<OUT_DIM;k_++){
                for(int l_=0;l_<OUT_DIM;l_++){
                    printf("%d\t",*(b + ((i/(ONE_BATCH_SIZE))*OUT_DIM*OUT_DIM*CHANNELS + k_*OUT_DIM*CHANNELS + l_*CHANNELS+(ONE_COl_IN_INDEX))));
                }
                printf("\n");
            }
            printf("\n");
            
            if(i == CHANNELS-1) return false;
            // return false;
        }
    return true;
}

void init_random(elem_t * buf, int len) {
    elem_t i = 0;
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;

      *ptr = (rand() % 5) - 2;
       //   *ptr = 1;
    }
}

void init_random_acc(acc_t * buf, int len) {
    elem_t i = 0;
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
      *ptr = (rand() % 5) - 2;
    }
}

void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    gemmini_flush(0);

    printf("Output dimension: %u\n\n", OUT_DIM);

    static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][CHANNELS] row_align(1);
    static elem_t weights[CHANNELS][KERNEL_DIM][KERNEL_DIM] row_align(1);
    static acc_t bias[CHANNELS] row_align_acc(1);
    static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][CHANNELS] row_align(1);

    printf("BATCH_SIZE : %d, IN_DIM : %d, CHANNELS : %d, KERNEL_DIM : %d, OUT_DIM : %d, PADDING : %d, STRIDE : %d\n"
    ,BATCH_SIZE,IN_DIM,CHANNELS,KERNEL_DIM,OUT_DIM,PADDING,STRIDE);
    if(NO_BIAS) printf("** NO BIAS ** \n");
    else printf("** BIAS EXISTS **\n");

    printf("Randomize inputs...\n");
    init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));
        
    printf("Randomize weights...\n");
    init_random(&weights[0][0][0], sizeof(weights) / sizeof(elem_t));
    
    printf("Randomize bias...\n");
    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
    else{
        init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
    }

    printf("CPU conv...\n");
    uint64_t start_cpu = read_cycles();

#ifndef SKIP_CPU
    tiled_conv_dw_auto(BATCH_SIZE, IN_DIM, CHANNELS, OUT_DIM,
            STRIDE, PADDING, KERNEL_DIM,

            (elem_t*)input,
            (elem_t*)weights,
            (acc_t*)bias,
            (elem_t*)output,

            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 1, 0, 0,

            CPU);
#endif

    uint64_t end_cpu = read_cycles();
    printf("CPU conv took %llu cycles\n", end_cpu - start_cpu);

    static elem_t output_mat[BATCH_SIZE][OUT_DIM][OUT_DIM][CHANNELS]; //row_align(1);
    static elem_t weights_shape_transform[CHANNELS][KERNEL_DIM][KERNEL_DIM]; // row_align(1);
    static elem_t weights_shape_transform_1d[CHANNELS][KERNEL_DIM*KERNEL_DIM]; // row_align(1);
    static elem_t weights_DWConv_V2[CHANNELS][DIM]; // row_align(1);

    /*

        weights_shape_transform : to control the order of weights stored in the scratchpad (~v1)
        weights_shape_transform_1d : to put 0 in the required position of the weights (~v2)
        weights_DWConv_V2 : weights that are finally used in DWConv v2 (~v2)
            -> The size is larger than the weights used in RiSA v1

    */
    // printf("transform weight`s shape...\n");
    for(int i=0;i<CHANNELS;i++){
        for(int k=KERNEL_DIM-1;k>=0;k--){
            for(int j=KERNEL_DIM-1;j>=0;j--){
                weights_shape_transform[i][KERNEL_DIM-1-k][KERNEL_DIM-1-j]=weights[i][j][k];
            }
        }
    }

    // printf("put 0 in the weight`s required position...\n");
    int idx=0;
    for(int i=0;i<CHANNELS;i++){
        for(int j=0;j<KERNEL_DIM;j++){
            for(int k=0;k<KERNEL_DIM;k++){
                weights_shape_transform_1d[i][idx] = weights_shape_transform[i][j][k];
                ++idx;
            }
        }
        idx=0;
    }

    // printf("make a weights_DWConv_V2...\r\n\n");
    int zero_num = 0;
    int weight_num = 0;
    int weight_idx = 0;

    int frame_height = DIM/KERNEL_DIM; //this should be matched with gemmini.h
    
    for(int i=0;i<CHANNELS;i++){
        for(int k=0;k<DIM;k++){
            if( k >= frame_height * KERNEL_DIM) {
                zero_num = 0; 
                weight_num = 0;
                weights_DWConv_V2[i][k] = 0;
            }
            else {
                if(zero_num < frame_height - KERNEL_DIM) {
                    weights_DWConv_V2[i][k] = 0;
                    ++zero_num;
                } else if(weight_num < KERNEL_DIM){
                    weights_DWConv_V2[i][k] = weights_shape_transform_1d[i][weight_idx];
                    ++weight_idx;
                    ++weight_num;
                }

                if(weight_num == KERNEL_DIM) {
                    zero_num = 0;
                    weight_num = 0;
                }
            }
        }
        zero_num=0;
        weight_num=0;
        weight_idx=0;
    }

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_dw_auto(BATCH_SIZE, IN_DIM, CHANNELS, OUT_DIM,
            STRIDE, PADDING, KERNEL_DIM,

            (elem_t*)input,
            (elem_t*)weights_DWConv_V2, //weights_DWConv_V2
            (acc_t*)bias,
            (elem_t*)output_mat,

            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 1, 0, 0, 
                                                //pool_size, pool_stride, pool_padding
            WS);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);


    assert(sizeof(output_mat) == sizeof(output));

    bool success = vec_is_equal(&output[0][0][0][0], &output_mat[0][0][0][0], sizeof(output) / sizeof(elem_t));

    printf("vec_is_equal : %s\n",success? "true" : "false"); 
    return 0;
}

