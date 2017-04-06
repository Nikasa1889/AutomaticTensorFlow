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
char RQ_NAME[RQ_NAME_MAX];
char eq_buffer[EQ_MAX_SIZE + 1];
char rq_buffer[RQ_MAX_SIZE + 1];

//Prototypes
int eq_open_write();
int eq_open_read();
int eq_send(Req *req);
struct Req eq_receive();

int rq_open_write();
int rq_open_read();
int rq_send(Res *res);
struct Res rq_receive();


float requestEvaluator(char * program, int length){
void evaluateForever (float (*evaluateFunc)(char *program)){

int eq_open_write(){
  EQ = mq_open(EQ_NAME, O_WRONLY);
  if ((mqd_t)-1 == EQ){
    return -1;
  } else {
    return 0;
  }
}

int eq_open_read(){
  struct mq_attr EQ_ATTR;
  EQ_ATTR.mq_flags = 0;
  EQ_ATTR.mq_maxmsg = EQ_MAX_MSG;
  EQ_ATTR.mq_msgsize = EQ_MAX_SIZE;
  EQ = mq_open(EQ_NAME, O_CREAT | O_RDONLY, 0644, &EQ_ATTR);

  if ((mqd_t) -1 == EQ){
    return -1;
  } else {
    return 0;
  }
}
int eq_send(Req *req){
  int ret;
  //char str[msg_len + RQ_NAME_MAX + 1];
  //strcpy(str, RQ_NAME);
  //strcat(str, ",");
  //strcat()
  ret = mq_send(EQ, (const char *) &req, sizeof(req), 0);
  if (ret >= 0){
    return 0;
  } else {
    return -1;
  }
}

Req eq_receive(){
  Req request;
  ssize_t bytes_read;
  bytes_read = mq_receive(EQ, (char *) &request, sizeof(request), NULL);
  //TODO: what if bytes_read < 0?
  return request;
}

int rq_open_read(){
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


Res rq_receive(){
  ssize_t bytes_read;
  Res response;
  bytes_read = mq_receive(RQ, (char *) &response, RQ_MAX_SIZE, NULL);
  return response;
}

int rq_send(Res *res){
  int ret;
  ret = mq_send(RQ, (const char *) &res, sizeof(res), 0);
  if (ret >= 0){
    return 0;
  } else {
    return -1;
  }
}

int initMyRQName(){
  sprintf(RQ_NAME, "%s%ld",RQ_PREFIX,(long)getpid());
  return 0;
}

Req createRequest(char * program, int length){
  Req request;
  strcpy(request.rq_name, RQ_NAME);
  strcpy(request.program, program);
  return request;
}

Res createResponse(float result){
  Res response;
  response.result = result;
  return response;
}

float requestEvaluator(char * program, int length){
  int ret;
  float result;
  Req request;
  Res response;
  initMyRQName();
  ret = eq_open_write();
  request = createRequest(program, length);
  ret = eq_send(&request);
  ret = rq_open_read();
  response = rq_receive();
  mq_close(RQ);
  mq_unlink(RQ_NAME);
  mq_close(EQ);
  return response.result;
}

void evaluateForever (float (*evaluateFunc)(char *program)){
  Req request;
  Res response;
  int ret;
  float result;
  ret =  eq_open_read();
  while(1){
    request = eq_receive();
    strcpy(RQ_NAME, request.rq_name);
    rq_open_write();
    result = (*evaluateFunc) (request.program);
    response = createResponse(result);
    ret = rq_send(&response);    
  }
}
