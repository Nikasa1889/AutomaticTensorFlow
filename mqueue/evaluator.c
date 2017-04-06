#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <mqueue.h>
#include <unistd.h>
#include "common.h"

mqd_t EQ;

mqd_t RQ;
char *RQ_NAME;
char eq_buffer[EQ_MAX_SIZE + 1];
char rq_buffer[RQ_MAX_SIZE + 1];
int eq_open_write(){
  EQ = mq_open(EQ_NAME, O_WRONLY);
  if ((mqd_t)-1 == EQ){
    return -1;
  } else {
    return 0;
  }
}

int eq_send(char *buffer, int msg_len){
  int result;
  result = mq_send(EQ, buffer, msg_len, 0);
  if (result >= 0){
    return 0;
  } else {
    return -1;
  }
}

int rq_open_read(){
  sprintf(RQ_NAME, "%s%ld",RQ_PREFIX,(long)getpid());
  struct mq_attr RQ_ATTR;
  RQ_ATTR.mq_flags = 0;
  RQ_ATTR.mq_maxmsg = RQ_MAX_MSG;
  RQ_ATTR.mq_msgsize = RQ_MAX_SIZE;
  RQ = mq_open(RQ_NAME, O_CREAT | O_RDONLY, 0644, &RQ_ATTR);

  if ((mqd_t)-1 == RQ){
    return -1;  
  } else {
    return 0;
  }
}

float rq_read(){
  ssize_t bytes_read;
  float result;
  bytes_read = mq_receive(RQ, rq_buffer, RQ_MAX_SIZE, NULL);
  if (bytes_read >= 0){
    result = *(float*)&rq_buffer;
    return result;
  } else {
    return -1;
  }
}

float eq_evaluate(char *buffer, int msg_len){
  int ret;
  float result;
  ret = eq_open_write();
  ret = eq_send(buffer, msg_len);
  ret = rq_open_read();
  result = rq_read();
  mq_close(RQ);
  mq_unlink(RQ_NAME);
  mq_close(EQ);
  return result;
}
