/** \brief nn_error.h, error and warn information
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <exception>
#include <string>
/**
 * error exception class
 **/
class nn_error : public std::exception
{
public:
  explicit nn_error(const std::string &msg) : msg_(msg) {}
  const char *what() const throw() override { return msg_.c_str(); }

private:
  std::string msg_;
};

/**
 * warning class (for debug)
 **/
class nn_warn
{
public:
  explicit nn_warn(const std::string &msg) : msg_(msg)
  {
  }

private:
  std::string msg_;
  std::string msg_h_ = std::string("[WARNING] ");
};

/**
 * info class (for debug)
 **/
class nn_info
{
public:
  explicit nn_info(const std::string &msg) : msg_(msg)
  {
  }

private:
  std::string msg_;
  std::string msg_h = std::string("[INFO] ");
};

class nn_not_implemented_error : public nn_error
{
public:
  explicit nn_not_implemented_error(const std::string &msg = "not implemented")
      : nn_error(msg) {}
};
