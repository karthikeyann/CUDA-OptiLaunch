# ------------------------------------------------
# Makefile
#
# Author: Karthikeyan Natarajan github.com/karthikeyann
# Date  : 2014-03-29
#
# ------------------------------------------------

# project name (generate executable with this name)
TARGET   = test

CC	   = nvcc
# compiling flags here
CFLAGS   = -I.

LINKER   = nvcc
# linking flags here
LFLAGS   = -I. -lm

# change these to set the proper directories where each files shoould be
SRCDIR   = src
OBJDIR   = lib
BINDIR   = .

SOURCES  := $(SRCDIR)/OptimalKernelLaunch.cu $(SRCDIR)/test.cu #$(wildcard $(SRCDIR)/*.cu)
INCLUDES := $(SRCDIR)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
rm	   = rm -f


$(BINDIR)/$(TARGET): $(OBJECTS)
	@$(LINKER) -o $@ $(LFLAGS) $(OBJECTS)
	@echo "Linking complete!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@$(CC) $(CFLAGS) -c $< -o $@ -I $(INCLUDES)
	@echo "Compiled "$<" successfully!"

.PHONEY: clean
clean:
	@$(rm) $(OBJECTS)
	@echo "Cleanup complete!"

.PHONEY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)
	@echo "Executable removed!"
