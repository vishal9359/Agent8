// Example C++ file for testing the compiler

#include <iostream>
#include <vector>

int CreateVolume(const std::string& name, int size) {
    // Validate input
    if (name.empty()) {
        return -1;
    }
    
    if (size <= 0) {
        return -2;
    }
    
    // Allocate space
    int allocated = AllocateSpace(size);
    if (allocated < 0) {
        return -3;
    }
    
    // Update metadata
    UpdateMetadata(name, allocated);
    
    return 0;
}

int AllocateSpace(int size) {
    int space = 0;
    
    for (int i = 0; i < size; i++) {
        space += 1024;
    }
    
    if (space > 1000000) {
        return -1;
    }
    
    return space;
}

void UpdateMetadata(const std::string& name, int space) {
    std::cout << "Updating metadata for " << name 
              << " with space " << space << std::endl;
}

int ProcessItems(std::vector<int>& items) {
    int count = 0;
    
    for (auto item : items) {
        if (item > 0) {
            count += item;
        } else {
            break;
        }
    }
    
    switch (count % 3) {
        case 0:
            return count;
        case 1:
            return count * 2;
        default:
            return count * 3;
    }
}
