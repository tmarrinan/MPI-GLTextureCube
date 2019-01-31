############
CXX= mpic++
CXX_FLAGS= -std=c++11 -DASIO_STANDALONE

MACHINE= $(shell uname -s)

NETSOCKET_DIR= $(HOME)/local
OPENSSL_DIR=/usr/local/opt/openssl
PXSTREAM_DIR= $(HOME)/Dev/pxstream

ifeq ($(MACHINE),Darwin)
	INC= -I/usr/local/include -I$(HOME)/local/include -I$(NETSOCKET_DIR)/include -I$(OPENSSL_DIR)/include -I/$(PXSTREAM_DIR)/include -I./include
	LIB= -L/usr/local/lib -L$(HOME)/local/lib -L$(NETSOCKET_DIR)/lib -L$(OPENSSL_DIR)/lib -L/$(PXSTREAM_DIR)/lib -lglfw -lglad -lnetsocket -lssl -lcrypto -lpthread -lpxstream
else
	INC= -I/usr/include -I/$(HOME)/Dev/pxstream/include -I./include
	LIB= -L/usr/lib64 -L$(NETSOCKET_DIR)/lib -L$(OPENSSL_DIR)/lib -L/$(PXSTREAM_DIR)/lib -lGL -lglfw -lglad -lnetsocket -lssl -lcrypto -lpthread -lpxstream
endif

SRCDIR= src
OBJDIR= obj
BINDIR= bin

OBJS= $(addprefix $(OBJDIR)/, main.o)
EXEC= $(addprefix $(BINDIR)/, texturecube)

mkdirs:= $(shell mkdir -p $(OBJDIR) $(BINDIR))


# BUILD EVERYTHING
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $^ $(LIB)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INC)


# REMOVE OLD FILES
clean:
	rm -f $(OBJS) $(EXEC)
