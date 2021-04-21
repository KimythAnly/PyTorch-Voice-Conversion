import argparse
import re as _re
import sys as _sys
from gettext import gettext as _


SUPPRESS = '==SUPPRESS=='
OPTIONAL = '?'
ZERO_OR_MORE = '*'
ONE_OR_MORE = '+'
PARSER = 'A...'
REMAINDER = '...'
_UNRECOGNIZED_ARGS_ATTR = '_unrecognized_args'

class _ChoicesPseudoAction(argparse.Action):
    def __init__(self, name, aliases, help):
        metavar = dest = name
        if aliases:
            metavar += ' (%s)' % ', '.join(aliases)
        super().__init__(option_strings=[], dest=dest, help=help,
                    metavar=metavar)
    def __len__(self):
        return len(self.metavar)
    
    def __str__(self):
        return self.metavar


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_action(self, action):
        if isinstance(action, str):
            if action:
                # determine the required width and the entry label
                help_position = min(self._action_max_length + 2,
                                    self._max_help_position)
                help_width = max(self._width - help_position, 11)
                action_width = help_position - self._current_indent - 2
                action_header = self._format_action_invocation(action)
                tup = self._current_indent, '', action_width, action_header
                action_header = '%*s%-*s  ' % tup
                parts = [action_header]
                return self._join_parts(parts)
        else:
            return super()._format_action(action)

    def _format_action_invocation(self, action):
        if isinstance(action, str):
            _action = _ChoicesPseudoAction(name=action, aliases=(), help='')
            return _action
        else:
            return super()._format_action_invocation(action)

    def _iter_indented_subactions(self, action):
        if not action.option_strings:
            if action.choices:
                if isinstance(action.choices, list):
                    self._indent()
                    yield from action.choices
                    self._dedent()

        try:
            get_subactions = action._get_subactions
        except AttributeError:
            pass
        else:
            self._indent()
            yield from get_subactions()
            self._dedent()
        


    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            # choice_strs = [str(choice) for choice in action.choices]
            # result = '<%s>' % ','.join(choice_strs)
            result = '<%s>' % action.dest
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = _('usage: ')

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(positionals + optionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:

                # break usage into wrappable parts
                part_regexp = (
                    r'\(.*?\)+(?=\s|$)|'
                    r'\[.*?\]+(?=\s|$)|'
                    r'\S+'
                )
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = _re.findall(part_regexp, opt_usage)
                pos_parts = _re.findall(part_regexp, pos_usage)
                assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = len(indent) - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent):]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + len(prog) + 1)
                    if opt_parts:
                        # lines = get_lines([prog] + opt_parts, indent, prefix)
                        # lines.extend(get_lines(pos_parts, indent))
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                        lines.extend(get_lines(opt_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    # parts = opt_parts + pos_parts
                    parts = pos_parts + opt_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        # lines.extend(get_lines(opt_parts, indent))
                        # lines.extend(get_lines(pos_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                        lines.extend(get_lines(opt_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '<%s>' % get_metavar(1)
        elif action.nargs == OPTIONAL:
            result = '[%s]' % get_metavar(1)
        elif action.nargs == ZERO_OR_MORE:
            result = '[%s [%s ...]]' % get_metavar(2)
        elif action.nargs == ONE_OR_MORE:
            result = '<%s> [%s ...]' % get_metavar(2)
        elif action.nargs == REMAINDER:
            result = '...'
        elif action.nargs == PARSER:
            result = '%s' % get_metavar(1)
        elif action.nargs == SUPPRESS:
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(action.nargs)]
            except TypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result

class CustomParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        kwargs['formatter_class'] = CustomHelpFormatter
        super().__init__(**kwargs)

    # =====================================
    # Command line argument parsing methods
    # =====================================
    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s')
            if args.module:
                subparser = self._subparsers._actions[1].choices[args.module]
                self.error(msg % ' '.join(argv), subparser=subparser)
            self.error(msg % ' '.join(argv))
        return args

    def error(self, message, subparser=None):
        """error(message: string)
        Prints a usage message incorporating the message to stderr and
        exits.
        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        if subparser:
            subparser.print_usage(_sys.stderr)
        else:
            self.print_usage(_sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, _('%(prog)s: error: %(message)s\n') % args)

def get_parser(**kwargs):
    return CustomParser(**kwargs)

# For debugging
if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('pos1')
    parser.add_argument('-o', '--opt1', help='A help.', type=str, default='1', dest='option1')
    parser.parse_args()



