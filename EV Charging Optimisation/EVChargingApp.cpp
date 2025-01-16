/*
I hereby certify that no other part of this submission has
been copied from any other sources, including the Internet,
books, or other student’s work, or generated from generative
AI tools, such as ChatGPT except the ones I have listed below:
// List the part of code you acquired from other resources
I hold a copy of this assignment that I can produce if the
original is lost or damaged.
*/

#include "Menu.h"

int main() {
    EVCharging charging;
    int choice;

    do {
        displayMenu();
        cin >> choice;
        handleMenuChoice(charging, choice);

        cout << "Do you want to perform another operation? (y/n): ";
        char another;
        cin >> another;
        if (another == 'n' || another == 'N') {
            break;
        }

    } while (true);

    return 0;
}
