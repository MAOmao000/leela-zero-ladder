/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "config.h"

#include <atomic>
#include <limits>
#include <string>

#ifdef _WIN32
#include <windows.h>
#define lockFile(file)                                       \
    HANDLE hFile = CreateFile(                               \
        file.c_str(),                                        \
        GENERIC_READ | GENERIC_WRITE,                        \
        FILE_SHARE_READ | FILE_SHARE_WRITE,                  \
        NULL,                                                \
        OPEN_ALWAYS,                                         \
        FILE_ATTRIBUTE_NORMAL,                               \
        NULL);                                               \
    if (hFile == INVALID_HANDLE_VALUE) {                     \
        myprintf_error("Could not lock file:%s.\n",          \
            file.c_str());                                   \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
    OVERLAPPED overlapped = {0};                             \
    auto locked = LockFileEx(                                \
        hFile,                                               \
        LOCKFILE_EXCLUSIVE_LOCK,                             \
        0,                                                   \
        0,                                                   \
        0,                                                   \
        &overlapped);                                        \
    if (!locked) {                                           \
        CloseHandle(hFile);                                  \
        myprintf_error("Could not lock file:%s.\n",          \
            file.c_str());                                   \
        exit(EXIT_FAILURE);                                  \
    }
#define unlockFile()                                         \
    if (!UnlockFileEx(hFile, 0, 0, 0, &overlapped)) {        \
        myprintf_error("Could not unlock file.\n");          \
    }                                                        \
    CloseHandle(hFile);
#else
#include <pwd.h>
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#define lockFile(file)                                       \
    int fd = open(file.c_str(), O_RDWR | O_CREAT, 0664);     \
    if (fd == -1) {                                          \
        myprintf_error("Could not lock file:%s.\n",          \
            file.c_str());                                   \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
    struct flock fl;                                         \
    fl.l_type = F_WRLCK;                                     \
    fl.l_whence = SEEK_SET;                                  \
    fl.l_start = 0;                                          \
    fl.l_len = 0;                                            \
    if (fcntl(fd, F_SETLKW, &fl) == -1) {                    \
        close(fd);                                           \
        myprintf_error("Could not lock file:%s.\n",          \
            file.c_str());                                   \
        exit(EXIT_FAILURE);                                  \
    }
#define unlockFile()                                         \
    fl.l_type = F_UNLCK;                                     \
    fl.l_whence = SEEK_SET;                                  \
    fl.l_start = 0;                                          \
    fl.l_len = 0;                                            \
    if (fcntl(fd, F_SETLK, &fl) == -1) {                     \
        myprintf_error("Could not unlock file.\n");          \
    }                                                        \
    close(fd);
#endif

#include "ThreadPool.h"

extern Utils::ThreadPool thread_pool;

namespace Utils {
    void myprintf_error(const char* fmt, ...);
    void myprintf(const char* fmt, ...);
    void gtp_printf(int id, const char* fmt, ...);
    void gtp_printf_raw(const char* fmt, ...);
    void gtp_fail_printf(int id, const char* fmt, ...);
    void log_input(const std::string& input);
    bool input_pending();

    template <class T>
    void atomic_add(std::atomic<T>& f, const T d) {
        T old = f.load();
        while (!f.compare_exchange_weak(old, old + d)) {}
    }

    template <typename T>
    T rotl(const T x, const int k) {
        return (x << k) | (x >> (std::numeric_limits<T>::digits - k));
    }

    inline bool is7bit(const int c) {
        return c >= 0 && c <= 127;
    }

    size_t ceilMultiple(size_t a, size_t b);

    std::string leelaz_file(const std::string& file);

    void create_z_table();
    float cached_t_quantile(int v);

    std::vector<float> softmax(const std::vector<float>& input,
                               const float temperature = 1.0f);
}

#endif
