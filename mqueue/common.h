#ifndef COMMON_H_
#define COMMON_H_

#define EQ_NAME     "/equeue"
#define EQ_MAX_SIZE    16384
#define EQ_MAX_MSG     5

#define RQ_PREFIX   "/rqueue"
#define RQ_MAX_SIZE 1
#define RQ_MAX_MSG 1

#define RQ_NAME_MAX 30
//#define MSG_STOP    "exit"

typedef struct Req
{
  char rq_name[RQ_NAME_MAX];
  char program[EQ_MAX_SIZE-20];
} Req;

typedef struct Res
{
  float result;
} Res;

#define CHECK(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "%s:%d: ", __func__, __LINE__); \
            perror(#x); \
            exit(-1); \
        } \
    } while (0) \


#endif /* #ifndef COMMON_H_ */
//Req buf;
//n = mq_receive(mqdes0, (char *) &buf, sizeof(buf), NULL);
//mq_send(mqdes1, (const char *) &buf, sizeof(buf), 0)
